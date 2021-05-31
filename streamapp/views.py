from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from keras.models import load_model
import cv2, os
import numpy as np
from django.conf import settings

face_detection_webcam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'haarcascade_frontalface_default.xml'))

maskNet = load_model(os.path.join(settings.BASE_DIR,'model.h5'))

class MaskDetect(object):
	def __init__(self):
		self.cap = cv2.VideoCapture(0)

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		while True:
			success, frame = self.cap.read()
			frameCopy = frame

			imgGray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
			faces = face_detection_webcam.detectMultiScale(imgGray,
											scaleFactor = 1.3,
											minNeighbors = 3,
											minSize = (100, 100),
											flags=cv2.CASCADE_SCALE_IMAGE)
			
			
			if len(faces)>0:
				preds = []
				locs = []
				for i in range(len(faces)):
					(x,y,w,h) = faces[i]
					face_img = frame[y:y+h , x:x+w]
					face_img=cv2.resize(face_img,(100,100))
					face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
					face_img =np.reshape(face_img,(1,100,100,1))
					face_img = face_img/255.0
					
					preds.append(maskNet.predict(face_img)[0])
					locs.append((x, y, x+w, y+h))

				for box, pred in zip(locs, preds):
					startX, startY, endX, endY = box
					withoutMask, mask = pred

					label = "With Mask" if mask > withoutMask else "No Mask"
					color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
					label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
					cv2.putText(frame, label, (startX, startY - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			ret, jpeg = cv2.imencode('.jpg', frame)
			return jpeg.tobytes()
		




def index(request):
	return render(request, 'home.html')

def predict(request):
	MaskDetect().__del__()
	return render(request, 'predict.html')

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def mask_feed(request):
	return StreamingHttpResponse(gen(MaskDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')
					