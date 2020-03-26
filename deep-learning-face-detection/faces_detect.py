import cv2
import imutils
import argparse
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to image")
ap.add_argument("-p","--prototxt",required=True,help="path to prototxt")
ap.add_argument("-m","--model",required=True,help='path to Caffeine Model')
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args=vars(ap.parse_args())	
print("Loading Model ....")
net=cv2.dnn.readNetFromCaffe(args['prototxt'],args['model'])
image=cv2.imread(args["image"])
(h,w)=image.shape[:2]
blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104,117,123))
net.setInput(blob)
detection=net.forward()
for i in range(0,detection.shape[2]):

	if detection[0,0,i,2]>args['confidence']:
		box=detection[0,0,i,3:7]*np.array([w,h,w,h])
		(x_start,y_start,x_end,y_end)=box.astype('int')
		cv2.rectangle(image,(x_start,y_start),(x_end,y_end),(0,0,255),2)
		text='{:.2f}'.format(detection[0,0,i,2]*100)
		y=y_start-10 if y_start-10>10 else y_start+10
		cv2.putText(image,text,(x_start,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
cv2.imshow("Output",image)
cv2.waitKey(0)
cv2.destroyAllWindows()		