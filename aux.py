import cv2

from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
import util




from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util



#Util vars
w = 1280
h = 720

n_video = "3"
intervalos = []
#abre video
if len(sys.argv) > 1:
	n_video =  sys.argv[1]
	if n_video== '2':
		cap = cv2.VideoCapture("video2.mp4")
	elif n_video == '3':
		cap = cv2.VideoCapture("video3.mp4")
	elif n_video == '1':
		cap = cv2.VideoCapture("video.mp4")
	else:
		cap = cv2.VideoCapture("video3.mp4")
else:
	cap = cv2.VideoCapture("video3.mp4")

with open("video"+n_video+"_intervalos.txt") as arquivo:
	linhas = arquivo.read().split("\n")
	for linha in linhas:
		if linha:
			valores = linha.split(":")
			intervalos.append([int(valores[0]), int(valores[1])])


inicio = False
eventDetect = util.Event_detect()

#cv2.VideoWriter_fourcc(*'XVID')

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w,h))
	
def update_objects(result, img, research):
	#global inicio

	(boxes, scores, classes, num) = result
	listaAux = []
	i = 0
	for x1, y1, x2, y2 in boxes[0]:
		if x1 == 0.:
			break
		listaAux.append([np.int32(x1*h), np.int32(x2*h) , np.int32(y1*w), np.int32(y2*w), classes[0][i]])
		
		i+=1

	eventDetect.atualiza_list(listaAux, img, research)










#Tensorflow Model Load

MODEL_NAME = 'traves'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'label_map.pbtxt')
NUM_CLASSES = 3


detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
 

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

frameCont = 0




flag = True
#Detection
int_atual = 0
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		try:
			print("proximo inntervalo: " + str(intervalos[int_atual][0]) + " ate " + str(intervalos[int_atual][1]))
			while True:
				ret, image_np = cap.read()
				if ret == True:
					frameCont = frameCont + 1
					if frameCont >= intervalos[int_atual][0]:
						if frameCont >= intervalos[int_atual][1]:
							int_atual+=1
							if int_atual == len(intervalos):
								print(str(eventDetect.aglu), str(eventDetect.nReconhecidos), str(eventDetect.totalJogadores), str(eventDetect.bola.predTotal), str(eventDetect.bola.ok))
								break
							print("proximo intervalo atual: " + str(intervalos[int_atual][0]) + " ate " + str(intervalos[int_atual][1]))
							flag = True
							with open("checkPoint.txt", "w") as auxFile:
								auxFile.write(str(eventDetect.ids) + "\n" + str(eventDetect.posses[0]) + "\n" + str(eventDetect.posses[1]))
							with open("checkPoint2.txt", "w") as auxFile:
								auxFile.write("Aglutinacoes: " + str(eventDetect.aglu)+ " \n")
								auxFile.write("nao reconhecidos: " + str(eventDetect.nReconhecidos)+ " \n")
								auxFile.write("total jogadores reconhecidos: " + str(eventDetect.totalJogadores)+ " \n")
								auxFile.write("total bola perdidas: " + str(eventDetect.bola.predTotal)+ " \n")
								auxFile.write("total bola achada: " + str(eventDetect.bola.ok)+ " \n")
								auxFile.write("kmeans erros: " + str(util.globalErro)+ " \n")
								auxFile.write("kmeans Acertos: " + str(util.acerto) + " \n")
								auxFile.write("dominios: " + str(eventDetect.totalDominios) + " \n")
							#input()
						else:			
							print(frameCont)
							# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
							image_np_expanded = np.expand_dims(image_np, axis=0)

							image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
							# Each box represents a part of the image where a particular object was detected.
							boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
							# Each score represent how level of confidence for each of the objects.
							# Score is shown on the result image, together with the class label.
							scores = detection_graph.get_tensor_by_name('detection_scores:0')
							classes = detection_graph.get_tensor_by_name('detection_classes:0')
							num_detections = detection_graph.get_tensor_by_name('num_detections:0')
							# Actual detection.
							#(boxes, scores, classes, num_detections) 
							result = sess.run(
									[boxes, scores, classes, num_detections],
									feed_dict={image_tensor: image_np_expanded})
							# vis_util.visualize_boxes_and_labels_on_image_array(
							# 		image_np,
							# 		np.squeeze(result[0]),
							# 		np.squeeze(result[2]).astype(np.int32),
							# 		np.squeeze(result[1]),
							# 		category_index,
							# 		use_normalized_coordinates=True,
							# 		line_thickness=8)
							#print(frameCont)
							
							update_objects(result, image_np, flag)
							eventDetect.draw_objects(image_np, frameCont)
							eventDetect.detect(image_np, frameCont, flag)

							#out.write(img)
							#cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
							out.write(image_np)
							#out.write(image_np)
							#out.write(image_np)
							#out.write(image_np)
							#out.write(image_np)
							#out.write(image_np) 
							#cv2.imwrite('frames/img_j3_'+str(frameCont) +".png",image_np)
							#cv2.imshow('object detection', cv2.resize(image_np, (w,h)))
							if cv2.waitKey(25) & 0xFF == ord('q'):
								break
							flag = False
							print("==============================================================")
					if frameCont%1000 == 0:
						print(frameCont)
				else:
					break
		except KeyboardInterrupt:
			out.release()
			cv2.destroyAllWindows()

print("Aglutinacoes: " + str(eventDetect.aglu))
print("nao reconhecidos: " + str(eventDetect.nReconhecidos))
print("total jogadores reconhecidos: " + str(eventDetect.totalJogadores))
print("total bola perdidas: " + str(eventDetect.bola.predTotal))
print("total bola achada: " + str(eventDetect.bola.ok))
print("kmeans erros: " + str(util.globalErro))
print("kmeans Acertos: " + str(util.acerto))

