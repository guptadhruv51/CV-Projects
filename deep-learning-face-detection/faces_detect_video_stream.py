import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import time
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-p','--prototxt',required=True,help='path to prototxt')
ap.add_argument('-m','--model',required=True,help='path to model')
args=vars(ap.parse_args())
net=cv2.dnn.readNetFromCaffe(args['prototxt'],args['model'])
vs=VideoStream(src=0).start()
time.sleep(4.0)
while True:
	frame=vs.read()

	frame=imutils.resize(frame,width=400)
	(h,w)=frame.shape[:2]
	blobs=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104,177,203))
	net.setInput(blobs)
	detections=net.forward()
	for i in range(detections.shape[2]):
		
		if detections[0,0,i,2]<0.5:
			continue
		
		box=detections[0,0,i,3:7]*np.array([w,h,w,h])

		(start_x,start_y,end_x,end_y)=box.astype('int')

		cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),(0,0,255),2)

		text="{:.2f}".format(detections[0,0,i,2]*100)

		y=start_y-10 if start_y-10>10 else start_y+10

		cv2.putText(frame,text,(start_x,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

	cv2.imshow("Frame",frame)

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()	

