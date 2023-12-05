import cv2
import face_recognition


def find_face_encodings(image_path):
	image = cv2.imread(image_path)
	face_enc = face_recognition.face_encodings(image)
	return face_enc[0]

def getFace(ine):
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
	gray = cv2.cvtColor(ine, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	x = faces[0][0]
	y = faces[0][1]
	w = faces[0][2]
	h = faces[0][3]

	#print(faces)
	#print("",x, y, w, h)
	return ine[y:y+h, x:x+w], w, h
	for (x, y, w, h) in faces:
		cv2.rectangle(ine, (x, y), (x + w, y + h), (255, 0, 0), 5)
		roi_gray = gray[y:y+w, x:x+w]
		roi_color = ine[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

	


def getMatch(template, picture) :
	#soccerImg = cv2.imread("assets/soccer_practice.jpg", 0)
	#template = cv2.imread("assets/ball.png", 0)
	h, w = template.shape
	cv2.imshow("template", template)
	cv2.imshow("picture", picture)

	methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
	            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

	for method in methods:
		img = picture.copy()
		result = cv2.matchTemplate(img, template, method)
		#biggest value
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

		print(cv2.minMaxLoc(result))
		if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
			location = minLoc
		else: 
			location = maxLoc

		bottomRight = (location[0] + w, location[1] + h)
		cv2.rectangle(img, location, bottomRight, (0,255,0), 10)
		cv2.imshow("Match", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()




ine = cv2.imread("assets/ine.png", cv2.IMREAD_COLOR)
picture = cv2.imread("assets/person1.jpg", cv2.IMREAD_COLOR)

ineFace, w, h = getFace(ine)
pictureFace, w,h = getFace(picture)

#ineFace = cv2.resize(ineFace, (w, h))

#cv2.imshow("Ine",  ine)
#cv2.imshow("IneFace", ineFace)q
#cv2.imshow("Picture", pictureFace)

#print(ineFace.shape)
getMatch(cv2.cvtColor(ineFace, cv2.COLOR_BGR2GRAY), cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY))

#cv2.imshow("Face", ineFace)
cv2.waitKey(0)
cv2.destroyAllWindows()


image1 = find_face_encodings("assets/picture6.jpeg")
image2 = find_face_encodings("assets/picture3.jpg")

is_Same = face_recognition.compare_faces([image1], image2)[0]
print("IsSame: ",is_Same)
distance = face_recognition.face_distance([image1], image2)
distance = round(distance[0] * 100)
    
# calcuating accuracy level between images
accuracy = 100 - round(distance)    
#print("The images are same")
print(f"Accuracy Level: {accuracy}%")