import cv2
import torch
import numpy as np
from tracker import *


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('highway.mp4')

count=0
tracker = Tracker()
area1=[(446,446),(429,471),(528,472),(527,454)]
area2=[(593,410),(603,431),(659,425),(653,410)]
area3=[(361,439),(344,456),(410,458),(427,440)]
area4=[(694,406),(700,417),(733,414),(722,399)]



def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)
up=set()
up1=set()
down=set()
down1=set()
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),1)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),1)
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(255,0,0),1)
    cv2.polylines(frame,[np.array(area4,np.int32)],True,(255,0,0),1)

    results=model(frame)
#    frame=np.squeeze(results.render())
    list=[]
    for index,row in results.pandas().xyxy[0].iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        b=str(row['name'])
        if 'car' in b:
            list.append([x1,y1,x2,y2])
    idx_bbox=tracker.update(list)
    for x,y,w,h,id in idx_bbox:          
        cv2.rectangle(frame,(x,y),(w,h),(0,0,255),1)
        cv2.putText(frame,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((w,h)),False)
        result1=cv2.pointPolygonTest(np.array(area2,np.int32),((x,h)),False)
        result2=cv2.pointPolygonTest(np.array(area3,np.int32),((x,h)),False)
        result3=cv2.pointPolygonTest(np.array(area4,np.int32),((w,h)),False)
        if result >0:
            up.add(id)
        if result1 >0:
            down.add(id)
        if result2 >0:
            up1.add(id)
        if result3 >0:
            down1.add(id)      
    a1=len(up)
    cv2.putText(frame,str(a1),(541,461),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    a2=len(down)
    cv2.putText(frame,str(a2),(571,422),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    a3=len(up1)
    cv2.putText(frame,str(a3),(339,440),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    a4=len(down1)
    cv2.putText(frame,str(a4),(735,402),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()