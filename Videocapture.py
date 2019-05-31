# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:07:37 2019

@author: nvidia
"""

import cv2
import time
from detector import FaceDetector
import threading as td  
import queue
import os

import sys


        
def camera(qtask):
    qt = qtask
    cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    while cap.isOpened():
         ret, Frame = cap.read()
         if ret==True:
            if qt.empty():
                 qt.put(Frame)
         else:
             continue
        
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break

    cap.release()
    cv2.destroyAllWindows()
        
class detectface():
    def __init__(self,thresh):
           
        face_detector = FaceDetector()
        self.cnt = 0
        foldcnt = 0
        frame = []

        ##create fold
        Job_number = input('\n \n \n Please input the Employee No , (ex.48628) = ')        
        
        for dir_item in os.listdir('./data/'):
            foldcnt = foldcnt + 1   
        self.fold = str(foldcnt)        
        if not os.path.exists('./data/'+ self.fold):
            os.makedirs('./data/'+ self.fold)
        
        # data log 
        
#==============================================================================
#         if not os.path.exists("./Person_log.xls"):
#             cw =  open("./Person_log.csv",'w')
#             column="ID"+","+"工號\n"
#             cw.write(column)
#==============================================================================
        
        cw = open("./Person_log.csv",'a+')
        cw.write(self.fold +","+str(Job_number)+"\n")
            
  
        #Camera open    
        self.q = queue.Queue(2)
        self.cap=td.Thread(target=camera,args=(self.q,))
        self.cap.start()
        time.sleep(10)
        print("face detect start")
        now = time.time()
        
        while True:
            if not self.q.empty():
            
                now = time.time()           
                frame = self.q.get()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)           
                if thresh:
                    bboxes = face_detector.predict(rgb_frame, thresh)
                else:
                    bboxes = face_detector.predict(rgb_frame)
    
                ann_frame = self.annotate_image(frame, bboxes)
                
                if cv2.waitKey(1) & 0xFF == ord('d'):
                    cv2.imwrite('./sample.jpg',ann_frame)
                
                
    
                cv2.imshow('window', ann_frame)
                print("FPS: {:0.2f}".format(1 / (time.time() - now)), end="\r", flush=True)
               
    def annotate_image(self,frame, bboxes):
            ret = frame[:]
        
            img_h, img_w, _ = frame.shape
            
            
            for x, y, w, h, p in bboxes:
                cv2.rectangle(ret, (int(x - 2*w/3), int(y - 3*h/4)), (int(x + 2*w/3), int(y + 3*h/4)), (0, 255, 0), 2) 
                fstore = ret[int(y - h):int(y + h),int(x - w):int(x + w)]
                if self.cnt>=500 :
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(ret,'Finish',(x ,y ),font,1,(100,0,255),4)
                    cv2.destroyAllWindows()
                    sys.exit("\n \n \n \n Please close the Windows \n  ")    
                   
                else:
                
                    self.cnt += 1
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(ret,'cnt:%d'%(self.cnt),(int(x+ 1.5*w) ,y ),font,1,(100,0,255),2)
#                    cv2.imwrite('./data/'+'%s'%str(self.fold)+'/%s.jpg'%str(self.cnt) , fstore)             
            return ret
                
       
if __name__ == "__main__":
    thresh = 0.85
    df = detectface(thresh)
 

    
    
    
