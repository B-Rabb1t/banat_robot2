#!/usr/bin/env python3
from __future__ import print_function


import rospy
import cv2
from time import sleep
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
#from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
import math
from FSX_s import FSX
#from simple_pid import PID
#from parse import citireTXT
roll = pitch = yaw = 0.0
directie=[1,1,0,1,0]
nr= [3,2,1,0,1]
def miscareTask2(directie,nr):  
  if directie==1:
    self.command.linear.x=0.1*nr
    self.command.angular.z=0.4   #intoarcere mid intoarcere
    self.pub.publish(self.command)
  elif directie==0:
    self.command.linear.x=0.1*nr
    self.command.angular.z=-0.4   #intoarcere mid intoarcere
    self.pub.publish(self.command)
class image_converter:
  
  def __init__(self):
    #self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/front/image_raw",Image,self.callback)
    #self.odom_sub=rospy.Subscriber("/odometry/filtered",Odometry,self.get_rotation)
    self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    self.command=Twist()
    #self.pid = PID(1.2, 0.2, 0.1)
    #self.pid.sample_time = 0.1
    self.intoarcere=0
    self.decizie=0
    self.decizie1=0
    self.ok=0
    self.contorPentruOK=0
    self.contorGreen=0
    self.okdeintoar=0
    self.contorLane=0
  def callback(self,data):
    
    try:
      intoar=0
      time=rospy.Time()
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") #frameul
      fs=FSX(cv_image,50)
      fs.PrepairFSV()
      frame,angle,accel,intoar,mid=fs.ClassifyFSV()
      target_angle=angle-90
      
      #print(theta,target_angle* math.pi/180 ,yaw )
      target_rad =target_angle * math.pi/180 #targhet angle il scoti tu
      
      if  (intoar or mid):# ori intoarcere ori mid ca plm te poti astepta la orice fel de noise
        if self.ok!=2 and intoar==1 :
          self.ok=1
        
        
          
       
          ###intoarcere
        #cv2.putText(frame,"Intoarcere",(320,240),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),2,cv2.LINE_AA)
          
       # self.decizie=self.decizie+1
        #if self.decizie%2==1:


            
         # self.intoarcere=0
          #self.ref_yaw=yaw*180/math.pi
          #print(abs(self.ref_yaw-yaw))
        

        #if intoar: 
          #self.command.linear.x=0.1
          #self.command.angular.z=0.4
          #self.pub.publish(self.command)
       
        #sleep(15)
          
        if(self.ok==1 and self.okdeintoar==1):################# GOING FORWARD A BIT ONLY ONCE ######################################
          cv2.putText(frame,"forward I+c:"+str(self.contorPentruOK),(320,240),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),2,cv2.LINE_AA)
          self.contorPentruOK=self.contorPentruOK+1
          if self.contorPentruOK>6:
            if self.okdeintoar==1:
              self.decizie=directie[self.contorLane]
              self.nr=nr[self.contorLane]
              self.contorLane=self.contorLane+1
              
          
              self.okdeintoar=0#timpul de mers in fata 0.1ms~10fps
            self.contorGreen=0
            self.ok=2
            #self.okdeintoar=0
            if directie[self.contorLane]==1:
              self.command.linear.x=0.1*nr[self.contorLane]
              self.command.angular.z=0.4   #intoarcere mid intoarcere
              self.pub.publish(self.command)
            elif directie[self.contorLane]==0:
              self.command.linear.x=0.1*nr[self.contorLane]
              self.command.angular.z=-0.4   #intoarcere mid intoarcere
              self.pub.publish(self.command)
            #cv2.putText(frame,"ROtate1",(320,240),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),2,cv2.LINE_AA)
          else:
            self.command.linear.x=1.1      #intoar prima data sa mearga in fata primele 5 iteratii
            self.command.angular.z=0
            self.pub.publish(self.command)  
        elif mid and self.okdeintoar==1 and self.ok==2: #################### MID INTOARCERE CAND VEDE si frunze da il doare in pula
            #selfself..ok=2
          cv2.putText(frame,"mid",(320,240),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),2,cv2.LINE_AA)
          if directie[self.contorLane]==1:
            self.command.linear.x=0.1*nr[self.contorLane]
            self.command.angular.z=0.4   #intoarcere mid intoarcere
            self.pub.publish(self.command)
          elif directie[self.contorLane]==0:
            self.command.linear.x=0.1*nr[self.contorLane]
            self.command.angular.z=-0.4   #intoarcere mid intoarcere
            self.pub.publish(self.command)
          else: 
            if directie[self.contorLane]==1:
              self.command.linear.x=0.1*nr[self.contorLane]
              self.command.angular.z=0.4   #intoarcere mid intoarcere
              self.pub.publish(self.command)
            elif directie[self.contorLane]==0:
              
              self.command.linear.x=0.1*nr[self.contorLane]
              self.command.angular.z=-0.4   #intoarcere mid intoarcere
              self.pub.publish(self.command)
              #self.ok=1
              #self.contorGreen=0
            #if self.decizie==1:
        else:
          if directie[self.contorLane]==1:
            self.command.linear.x=0.1
            self.command.angular.z=0.4   #intoarcere mid intoarcere
            self.pub.publish(self.command)
          elif directie[self.contorLane]==0:
            self.command.linear.x=0.1
            self.command.angular.z=-0.4   #intoarcere mid intoarcere
            self.pub.publish(self.command)
          
        
            
            
        
          
          
      else:
        if self.ok==2: ############### PRE-LANE (pozitionare ######################
          cv2.putText(frame,"Prelane",(320,240),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),2,cv2.LINE_AA)
          if (fs.StangaLim>fs.width/2-30 or fs.StangaLim1> fs.width/2-30 or fs.DreaptaLim<fs.width/2+30 or fs.DreaptaLim1<fs.width/2+30) and directie[self.contorLane]==1:
            self.okdeintoar=0
            #self.decizie=1
            self.command.linear.x=0.1 #si daca se fute intoarcerea , o redresam oleaca
            self.command.angular.z = 0.4
            self.pub.publish(self.command)
          elif(fs.StangaLim>fs.width/2-30 or fs.StangaLim1> fs.width/2-30 or fs.DreaptaLim<fs.width/2+30 or fs.DreaptaLim1<fs.width/2+30) and directie[self.contorLane]==0:
            #self.decizie=0  
            self.command.linear.x=0.1 #si daca se fute intoarcerea , o redresam oleaca
            self.command.angular.z = -0.4
            self.pub.publish(self.command)
          else:
            self.ok=0
            #self.contorPentruOK=0

        #self.ok=0
        else:
          cv2.putText(frame,"LANE+ C:"+str(self.contorGreen),(320,240),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),2,cv2.LINE_AA)
          self.contorPentruOK=0         ################ LANE #######################
          self.command.angular.z = 1*(target_rad-yaw) #unghiul la care sa se roteasa
          self.command.linear.x=10 #viteza de mers in fata
          self.pub.publish(self.command)
          self.contorGreen=self.contorGreen+1
          if self.contorGreen<30:
            self.okdeintoar=0
          if self.contorGreen>30:
            self.okdeintoar=1
          
            #contorPentruOK=0
            
          else:
            self.okdeintoar=0
          
          #if self.contorGreen>50:
            
            
              
      #print(self.command)
      print(self.decizie)
    except CvBridgeError as e:
      print(e)
    
    


    cv2.imshow("Image window", frame)
    key=cv2.waitKey(3)
    #if key:def get_rotation (self,msg):
    #glob
      #exit(0)
  #def get_rotation (self,msg):
  #  global roll, pitch, yaw
  #  q = msg.pose.pose.orientation
    #orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
  #  yaw = math.atan2(2.0*(q.y*q.z + q.w*q.x),+1.0 - 2.0*(q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z))
    #print(yaw*180/math.pi)
    #print yaw
  #def Command(self):
    

def main():
  rospy.init_node('image_converter', anonymous=True)

  ic = image_converter()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
