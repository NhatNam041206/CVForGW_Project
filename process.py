import cv2
import os
import datetime
import numpy as np

class CustomError(Exception):
    pass

def readImg(parentPath,log):
    if log: 
        print('<Log> Log enable!')
        print(f'<System> Reading images from {parentPath}.')

    imgs=[]
    edges_imgs=[]

    for path in os.listdir(parentPath):
        
        if log: print(f'<System> Image {path} from {os.path.join(parentPath,path)}')

        try:

            img=cv2.imread(os.path.join(parentPath,path))
            
            arrayTemp=np.array(img.shape[:-1])*0.4

            W,H=int(arrayTemp[0]),int(arrayTemp[1])

            img=cv2.resize(img,(H,W))

            # 1. Convert to gray & denoise
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 2. Outline extraction with Canny
            edges = cv2.Canny(blur, 10, 200)

            imgs.append(img)
            edges_imgs.append(edges)

        except Exception: print('<Error> Failed to read imgs, checked path again!')        

    print(f'<System> Reading and transforming complete!')

    return imgs, edges_imgs

def VideoProcessing(path, log=True):
    if log:
        print(f"[System] Reading video {path}")

    name       = os.path.basename(path)
    savedPath  = os.path.join("results", "vids", "FLines",name)
    savedPathEdges  = os.path.join("results", "vids", "Edges",name)
    savedPathGray  = os.path.join("results", "vids", "Gray",name)
    os.makedirs(os.path.dirname(savedPath), exist_ok=True)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise CustomError(f"Cannot open video {path}")

    # --- đọc khung đầu để biết kích thước ---
    ret, frame = cap.read()
    if not ret:
        raise CustomError("Video empty!")

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(savedPath, fourcc, 30, (w, h))
    edges_out = cv2.VideoWriter(savedPathEdges, fourcc, 30, (w, h))
    gray_out    = cv2.VideoWriter(savedPathGray, fourcc, 30, (w, h))
    if not out.isOpened():
        raise CustomError("Cannot open VideoWriter")

    while True:
        # <- dùng lại frame vừa đọc ở trên
        cv2.imshow("frame", frame)

        # (nếu cần resize, resize cả frame lẫn cấu hình writer
        # ngay từ ĐẦU, đừng resize lắt nhắt trong vòng lặp)

        # ---- xử lý ----
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 10, 200)
        cv2.imshow('Edges',edges)

        lines = linesGenerate(edges)

        gray_bgr  = cv2.cvtColor(gray,  cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        edges_out.write(edges_bgr)
        gray_out.write(gray_bgr)

        if lines is not None:
            filt  = filteredLines(lines, False, name)
            _, frame_drawn = drawLine(filt, frame, 3)
            out.write(frame_drawn)
            
        else:
            out.write(frame)

        # hiển thị 1 ms và bắt phím Esc thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # đọc khung tiếp
        ret, frame = cap.read()
        if not ret:
            break

    print(f"[System] Saved to {savedPath}")
    cap.release()
    out.release()
    gray_out.release()
    edges_out.release()
    cv2.destroyAllWindows()

def convertDegree(thetas):
    alpha=thetas - np.pi/2
    return np.abs(np.degrees(alpha))

def convertCoordinates(rho,theta):
    x0=rho*np.cos(theta)
    y0=rho*np.sin(theta)

    a=np.cos(theta)
    b=np.sin(theta)

    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))

    return (x1,y1),(x2,y2)

def drawLine(lines,img,thickness):
    img_c=img.copy()
    for i in range(len(lines)):
        rho=lines[i][0][0]
        theta=lines[i][0][1]
        
        (x1,y1),(x2,y2)=convertCoordinates(rho,theta)

        cv2.line(img_c,(x1,y1),(x2,y2),(255,0,0),thickness)
        return lines, img_c

def linesGenerate(edges):
    lines=[]
    lines=cv2.HoughLines(edges, 1,np.pi/180,240)
    if lines is None:
        return 

    return lines

def filteredLines(lines,log,name):

    if log:
        path=f'logs/filtered_lines_logging_{name}_.txt'
        f=open(path,'w+')       
        print(f'<Log> The results will be saved with the name of the given file. Destination: {path}')

        if not name:
            raise CustomError(f'<Error> Please input the name for logging process.')


    lines_filtered=[]
    visited=[]
    angleBias=0.3
    p_bias=20

    log_infs=f'Log on {name}.jpg image at {datetime.datetime.now()}'

    for i in range(len(lines)-1):
        lines_=[]

        check=False

        if list(lines[i][0]) in visited: continue

        for j in range(i,len(lines)):

            c=False

            angleD=np.abs(lines[i][0][1]-lines[j][0][1])
            pD=np.abs(lines[i][0][0]-lines[j][0][0])
            
            log_infs+=f'\nFirst line: {lines[i]}; Second line: {lines[j]}; Difference:\nAngle: {angleD}\nLine: {pD}\nCond: '

            if angleD<angleBias and pD<p_bias:
            
                log_infs+='Take\n'
            
                lines_.append(lines[i])
                lines_.append(lines[j])
                visited.append(list(lines[i][0]))
                visited.append(list(lines[j][0]))

                check=True
                c=True
                
            if not c:
            
                log_infs+='Pass\n'

        
        log_infs+='---Take and Average Process---\n'
        
        if not check:
            log_infs+=f'Not average; Value: {lines[i]}\n'
            
            lines_filtered.append(lines[i])
            visited.append(list(lines[i][0]))
            
            log_infs+=f'Visited: {visited}\n'
            log_infs+='-----------------------------\n'
        
        else:
            log_infs+=f'Average; value need to average {lines_}\n'
            log_infs+=f'Values: {np.mean(lines_,axis=0)}\n'
            
            lines_filtered.append(np.mean(lines_,axis=0))
            
            log_infs+=f'Visited: {visited}\n'
            log_infs+='-----------------------------\n'
    log_infs+=f'Result: {lines_filtered}\n'

    if log:
        f.write(log_infs)
        f.close()

    return lines_filtered