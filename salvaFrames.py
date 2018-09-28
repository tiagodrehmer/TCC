import cv2
cap = cv2.VideoCapture("video.mp4")
cont = 0
while cont < 80000:
	ret, img = cap.read()
	if cont > 10000:
		if cont % 400 == 0:
			cv2.imwrite('frames/img_'+str(cont) +".png",img)
			print(cont)
	cont+=1
	#print(cont)