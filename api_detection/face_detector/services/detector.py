import numpy as np
import os
import cv2
import dlib

FACE_DETECTOR_PATH = "{base_path}/../models/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
FACE_DETECTOR_DLIB_PATH = "{base_path}/../models/mmod_human_face_detector.dat".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

def haarcascade_frontalface_default(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    rects = [(int(x), int(y), int(x + w), int(y + h))
             for (x, y, w, h) in rects]
    return rects

def dlibDetector(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_DLIB_PATH)
    result = detector(image)
    rects = [convert_and_trim_bb(image, r.rect) for r in result]
    return rects

def convertFromBytes(data):
    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def convert_and_trim_bb(image, rect):
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)
