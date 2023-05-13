#pip install opencv
import cv2
import numpy as np 
#pip install cmake dlib face recognition
import face_recognition as fr
imgelon_bgr=fr.load_image_file('elon.jpg')
imgelon_rgb=cv2.cvtColor(imgelon_bgr,cv2.COLOR_BGR2RGB)
#--Finding face locations for drawing bounding boxes--
face=fr.face_locations(imgelon_rgb)[0]
copy=imgelon_rgb.copy()
#---Drawing the Rectange--
cv2.rectangle(copy,(face[3],face[0]),(face[1],face[2]),(255,0,255),2)
#--testing
train_encode=fr.face_encodings(imgelon_rgb)[0]
#--testing 
test=fr.load_image_file('elon2.jpg')
test=cv2.cvtColor(test,cv2.COLOR_BGR2RGB)
test_encode=fr.face_encodings(test)[0]
testing=fr.compare_faces([train_encode],test_encode)
print(testing)
#cv2.imshow('copy',copy)
#cv2.imshow('bgr',imgelon_bgr)
#cv2.imshow('rgb',imgelon_rgb)
#cv2.waitKey(0)


