from django.test import TestCase
import requests
import cv2

# define the URL to our face detection API
url = "http://127.0.0.1:8000/face_detection/detect"
# use our face detection API to find faces in images via image URL
image = cv2.imread("obama.jpg")
payload = {"image": image}
r = requests.post(url, data=payload).json()
print("obama.jpg: {}".format(r))
# loop over the faces and draw them on the image
for (startX, startY, endX, endY) in r["faces"]:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show the output image
cv2.imshow("obama.jpg", image)
cv2.waitKey(0)
# load our image and now use the face detection API to find faces in
# images by uploading an image directly
image = cv2.imread("adrian.jpg")
payload = {"image": open("adrian.jpg", "rb")}
r = requests.post(url, files=payload).json()
print("adrian.jpg: {}".format(r))
# loop over the faces and draw them on the image
for (startX, startY, endX, endY) in r["faces"]:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show the output image
cv2.imshow("adrian.jpg", image)
cv2.waitKey(0)
