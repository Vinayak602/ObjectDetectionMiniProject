# import the necessary packages

import numpy as np
import cv2

thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
cap = cv2.VideoCapture('D:\Mini Project Vinayak Gupta\Road_Traffic.gif')
cap.set(3,640) #width 
cap.set(4,488) #height 
 
# initialize the list of class labels MobileNet SSD was trained to 
# detect, then generate a set of bounding box colors for each class
classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()
print(classNames)

font = cv2.FONT_HERSHEY_PLAIN
#font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    if len(classIds) != 0:
        for i in indices:
            box = bbox[i]
            confidence = str(round(confs[i]*100,2))
            color = Colors[classIds[i]-1]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
            cv2.putText(img, classNames[classIds[i]-1]+" "+confidence+"%",(x+10,y+20),
                        font,1,color,2)

    # show the output img
    cv2.imshow("Output",img)
    key=cv2.waitKey(1)& 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break

cv2.destroyAllWindows()
    