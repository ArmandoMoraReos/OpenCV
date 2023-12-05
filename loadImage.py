import cv2

import numpy as np

cap = cv2.VideoCapture(0)
while True:
	ret , frame = cap.read()
	width = int(cap.get(3))
	height = int(cap.get(4))

	tmp = frame[100:200, 100:200]
	frame[0:100, 300:400] = tmp
	#frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
	img = cv2.line(frame, (0,0), (width, height), (255,0,0), 10)
	img = cv2.rectangle(img, (50,50), (100,100), (0,0,255), 10)
	#img = cv2.circle(img, (300, 300), 60, (255,255,0), -1)
	font = cv2.FONT_HERSHEY_SIMPLEX
	img = cv2.putText(img, "Hello opencv", (90,180), font, 1, (0,0,0), 5, cv2.LINE_AA)

	lowerBlue = np.array([100, 70, 70])
	upperBlue = np.array([130, 255, 255])
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
	mask = cv2.inRange(hsv, lowerBlue, upperBlue)

	result = cv2.bitwise_and(frame, frame, mask = mask)

	cv2.imshow("Video", result)
	cv2.imshow("Mask", mask)

	corners = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(corners, 100, 0.01, 11)
	corners = np.int0(corners)

	for corner in corners:
		x, y = corner.ravel()
		cv2.circle(img, (x, y), 5, (0,255,0), -1)

	cv2.imshow("Corners",img)
	print(frame.shape)

	if cv2.waitKey(1) == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()


img = cv2.imread("assets/galaxy.png", cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
print(type(img))
print(img.shape)
cv2.imwrite("test2.png", img)

#cv2.imshow("Practice1", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


img2 = cv2.imread("assets/chess2.png", cv2.IMREAD_COLOR)
img2 = cv2.resize(img2, (0, 0), fx = 2, fy = 2)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
for corner in corners:
	x, y = corner.ravel()
	cv2.circle(img2, (x, y), 5, (0,255,0), -1)


cv2.imshow("Corners2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


soccerImg = cv2.imread("assets/soccer_practice.jpg", 0)
template = cv2.imread("assets/ball.png", 0)
h, w = template.shape


methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
	img = soccerImg.copy()
	result = cv2.matchTemplate(img, template, method)
	#biggest value
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		location = minLoc
	else: 
		location = maxLoc

	bottomRight = (location[0] + w, location[1] + h)
	cv2.rectangle(img, location, bottomRight, 255, 5)
	cv2.imshow("Match", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
		roi_gray = gray[y:y+w, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

	cv2.imshow("frame", frame)
	if cv2.waitKey(1) == ord("q"):
		break

	cap.release()
	cv2.destroyAllWindows()