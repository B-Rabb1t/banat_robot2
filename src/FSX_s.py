import time
import math
import cv2
import numpy as np
from numpy import matrix
import imutils

     

#frame=cv2.imread("Track.png")
def getAngle(a, b, c):
  ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
  return ang + 360 if ang < 0 else ang
def distance(p1,p2):
  dist = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
  return dist
class FSX:
  
  
  
  def __init__(self,frame=None,sign=None):
    self.segments=50
    self.x_seg=[0]*50
    self.y_seg=[10]*50
    self.frame=frame
    self.sign=sign
    self.height=frame.shape[0]
    self.width=frame.shape[1]
    self.fsz = np.zeros([self.height,self.width,1],dtype=np.uint8)  
    self.B=[int(self.width/2),self.height]
    self.C=[self.width,self.height]
    self.font = cv2.FONT_HERSHEY_SIMPLEX
    self.StangaLim=0
    self.DreaptaLim=0
    self.mijloc=0
    self.FreeSpaceZone = np.zeros([self.height,self.width,1],dtype=np.uint8)
    self.overtake=0
    self.Over=[0,0]
    self.Re=[0,0]
    self.retake=0
    self.park=0
  def PrepairFSV(self):
    #mask=cv2.Canny(self.frame,100,90)
    kernel = np.ones((100,100), np.uint8)
    kernel2 = np.ones((3,3), np.uint8)
    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 60, 25), (70, 255,255))
    mask =cv2.erode(mask,kernel2)
    mask =cv2.dilate(mask,kernel2)
    mask = cv2.dilate(mask, kernel)
    cv2.imshow("mask",mask)
    #mask2=mask.copy()
    
    #kernel = np.ones((3,3),np.uint8)
    #kernel1 = np.ones((1,1),np.uint8)
    #lernel3 =np.ones((1,1),np.uint8)
    #mask=cv2.dilate(mask,kernel)
    #mask=cv2.threshold(mask,120,255,cv2.THRESH_BINARY)
   
    #mask2=cv2.dilate(mask2,kernel1)
    #mask2=cv2.erode(mask2,kernel1)
    y_form=[10]*self.segments
    x_form=[0]*self.segments    
    
    StangaLim=int(self.width/2)
    DreaptaLim=int(self.width/2)
    StangaLim1=int(self.width/2)
    DreaptaLim1=int(self.width/2)
    StangaLim2=int(self.width/2)
    DreaptaLim2=int(self.width/2)
    pic=np.array(mask)
    for i in range(int(self.width/2),self.width-10):
      if  pic[self.height-200][i]:
        DreaptaLim=i
        break
      
    for i in range(int(self.width/2)):
      if pic[self.height-200][int(self.width/2)-i]:
        StangaLim=int(self.width/2)-i
        break
    if(StangaLim==int(self.width/2)):
      StangaLim=1
    if(DreaptaLim==int(self.width/2)):
      DreaptaLim=self.width-1
    #-------------------------------------------------------------------------
    for i in range(int(self.width/2),self.width-10):
      if  pic[self.height-100][i]:
        DreaptaLim1=i
        break
    for i in range(int(self.width/2)):
      if pic[self.height-100][int(self.width/2)-i]:
        StangaLim1=int(self.width/2)-i
        break
    for i in range(int(self.width/2),self.width-10):
      if  pic[self.height-320][i]:
        DreaptaLim2=i
        break
    for i in range(int(self.width/2)):
      if pic[self.height-220][int(self.width/2)-i]:
        StangaLim2=int(self.width/2)-i
        break
    if(StangaLim==int(self.width/2)):
        StangaLim=1
    if(DreaptaLim==int(self.width/2)):
        DreaptaLim=self.width-1
    if(StangaLim1==int(self.width/2)):
        StangaLim1=1
    if(DreaptaLim1==int(self.width/2)):
        DreaptaLim1=self.width-1
    if(StangaLim2==int(self.width/2)):
        StangaLim2=1
    if(DreaptaLim2==int(self.width/2)):
        DreaptaLim2=self.width-1
    self.StangaLim=StangaLim
    self.DreaptaLim=DreaptaLim
    self.StangaLim1=StangaLim1
    self.DreaptaLim1=DreaptaLim1
    self.StangaLim2=StangaLim2
    self.DreaptaLim2=DreaptaLim2
    mijloc=int((DreaptaLim+StangaLim)/2)
    self.mijloc=mijloc
    #dl=DL(self.frame,mask,StangaLim,DreaptaLim)
    #self.Over[0],self.Over[1],self.overtake=dl.ProcessLeft()
    #self.Re[0],self.Re[1],self.retake=dl.ProcessRight()
   
    cv2.circle(self.frame,(StangaLim,self.height-200),5,(0,255,255),2)
    cv2.circle(self.frame,(DreaptaLim,self.height-200),5,(0,255,255),2)
    cv2.circle(self.frame,(StangaLim1,self.height-100),5,(0,255,255),2)
    cv2.circle(self.frame,(DreaptaLim1,self.height-100),5,(0,255,255),2)
    cv2.circle(self.frame,(StangaLim1,self.height-320),5,(0,255,255),2)
    cv2.circle(self.frame,(DreaptaLim1,self.height-320),5,(0,255,255),2)
    
    #cv2.imshow("canny",mask)
    tensor=0
    #pic2=np.array(mask2)
    for i in range(10):
      if int(mijloc-i*self.segments)+2*i*self.segments<DreaptaLim and int(mijloc-i*self.segments)+2*i*self.segments> StangaLim:
        tensor=i
    for i in range(self.segments):     
      y_form[i]=int(mijloc-tensor*self.segments)+2*tensor*i
    
    for i in range(40,self.height-150):
      for j in range(self.segments):
        if pic[i][y_form[j]]:
          if x_form[j]<i:
            x_form[j]=i
    for i in range(self.segments):
      self.x_seg[i]=x_form[i]
      self.y_seg[i]=y_form[i]
    
    



#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------    





  def ClassifyFSV(self):
    contorR=0
    R=0
    intersection=0
    contorG=0
    GyR=0
    GxR=0
    #
    x_cil=0
    y_cil=0
    angle=0
    for j in range(self.segments):
      if self.x_seg[j]>(self.height/2):
        contorR=contorR+1
        if j<self.segments-1 :
          GyR=GyR+self.x_seg[j]
          GxR=GxR+self.y_seg[j]
          R=R+1
          cv2.line(self.FreeSpaceZone,(self.y_seg[j+1],self.x_seg[j+1]),(self.y_seg[j],self.x_seg[j]),(255),1)
        else:
          if R>0.70*self.segments:
            intersection=1
            
      else:
        contorG=contorG+1
        x_cil=x_cil+self.y_seg[j]
        y_cil=y_cil+self.x_seg[j]
        if j<self.segments-1:
          cv2.line(self.FreeSpaceZone,(self.y_seg[j+1],self.x_seg[j+1]),(self.y_seg[j],self.x_seg[j]),(255),1)
    #int(GyR/R)
    cv2.line(self.FreeSpaceZone,(self.y_seg[0],self.x_seg[0]),(self.y_seg[0],self.height-10),(255),1)
    cv2.line(self.FreeSpaceZone,(self.y_seg[self.segments-1],self.x_seg[self.segments-1]),(self.y_seg[self.segments-1],self.height-10),(255),1)
    cv2.line(self.FreeSpaceZone,(self.y_seg[0],self.height-10),(self.y_seg[self.segments-1],self.height-10),(255),1)
    cnts = cv2.findContours(self.FreeSpaceZone, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    #print(intersection)
    cv2.drawContours(self.FreeSpaceZone, [c], 0, (255), -1)
    cv2.drawContours(self.frame,[c],0,(255,255,255),3)
    #img=cv2.imread("Data/grad1.jpg")
    #img_r=cv2.resize(img,(self.width,self.height),cv2.INTER_AREA)
    #mask_inv = cv2.bitwise_not(self.FreeSpaceZone)
   # img2_fg = cv2.bitwise_and(img_r,img_r,mask = self.FreeSpaceZone)
   # img1_bg = cv2.bitwise_and(self.frame,self.frame,mask = mask_inv)
   # self.frame = cv2.add(img1_bg,img2_fg)
    #cv2.imshow("grad",self.frame)
    okL=0
    okR=0
    
    
    
      
    X_arrow=int(sum(self.y_seg)/50)   #green 
    Y_arrow=int(sum(self.x_seg)/(50)) #green
    A=[X_arrow,Y_arrow]
    cv2.line(self.frame,(A[0],A[1]),(self.B[0],self.B[1]),(0,255,0),3)
    angle=getAngle(A,self.B,self.C)
        #angle=(angle+270)/4  
    
    angle=(angle+270)/4      
    cv2.rectangle(self.frame,(0,0),(100,60),(255,255,255),-1)
    cv2.rectangle(self.frame,(0,0),(100,60),(0,0,0),1)
    accel=0
    if self.mijloc>320:
      angle=angle-1
    if self.mijloc <320:
      angle=angle+1
    if angle<=100 and angle>1:
      accel=angle/100
    else:
      accel=angle/180
    #angle=90-angle
    intoar=0
    intoarMidintoar=0  
    if self.StangaLim==1 and self.DreaptaLim==self.width-1 and self.StangaLim1==1 and self.DreaptaLim1==self.width-1 and self.StangaLim2==1 and self.DreaptaLim2==self.width-1:
     # cv2.rectangle(self.frame,((int(self.width/2),int(self.height/2)),(int(self.width/2+100),int(self.height/2)-10),(255,255,255),-1)
      intoar=1
    if (self.StangaLim==1 and self.StangaLim1==1 and self.StangaLim2==1 )  ^ (self.DreaptaLim==self.width-1 and self.DreaptaLim1==self.width-1 and self.DreaptaLim2==self.width-1):
     # cv2.rectangle(self.frame,((int(self.width/2),int(self.height/2)),(int(self.width/2+100),int(self.height/2)-10),(255,255,255),-1)
      
      intoarMidintoar=1   
    #accel=accel-0.2
    cv2.putText(self.frame,"D: "+str(self.DreaptaLim),(5,11),self.font, 0.4,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(self.frame,"S: "+str(self.StangaLim),(5,23),self.font, 0.4,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(self.frame,"Angle: "+str(int(angle)),(5,35),self.font, 0.4,(0,0,0),1,cv2.LINE_AA)
    return self.frame,angle,accel,intoar,intoarMidintoar

    #cv2.imshow("FSZ",FreeSpaceZone)
