import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
import operator

w = 1280
h = 720
amarelo = (0,255,255)
vermelho = (0,0,255)
azul = (255,0,0)
branco = (255,255,255)
preto = (0, 0, 0)
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (640,360)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
globalErro = 0
acerto = 0
globalJogadores = 0
erroBola = 0
#cv2.putText(img, str(total) + " " + str(len(self.listaJAux)), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


class Jogador:
	def __init__(self, x1, x2, y1, y2, time, ide):
		self.y1 = y1
		self.y2 = y2
		self.cont=1
		if time >= 0:
			self.mediaTime = time
			self.contTime = 6
			self.time = time + 1
		else:
			self.mediaTime = 0
			self.contTime = 0
			self.time = 0
		self.vel = 0
		self.histVel = []
		self.histAltura = []
		self.flagAlt = True
		self.flagVel = True
		self.colisao = 0
		self.x2 = x2
		self.x1 = x1
		self.altura = x2 - x1
		self.id = ide
		self.pred = 0
		self.lastTime = 0
		self.colisor = None


	def atualiza(self, x1, x2, y1, y2, time, flag):
		self.atualiza_vel(y2)
		self.y1 = y1
		self.y2 = y2
		self.cont+=1
		self.atualiza_altura(x2, x1)
		if flag:
			self.time = time + 1
			self.lastTime = time + 1
			self.colisor = None

		else:
			if self.time == 0 and time >= 0 and self.colisao == 0:
				if self.contTime <= 7:
					self.mediaTime += time
					self.contTime += 1
				else:
					self.time = math.ceil(self.mediaTime/self.contTime) + 1
					print(str(self.id), str(self.time))
					#x = input()
					# if x == "1":
					# 	self.time = (self.time%2) + 1
					# 	global globalErro
					# 	globalErro += 1
					# elif x == "2":
					# 	self.time = (self.time%2) + 1
					# elif x != "3":
					# 	global acerto
					# 	acerto += 1
					# if self.colisor:
					# 	self.colisor.time = (self.time%2) + 1
					# 	self.colisor.lastTime = (self.time%2) + 1
					# 	self.colisor = None
					global acerto
					acerto += 1
					self.lastTime = self.time
				if self.contTime == 7 and not self.lasTime:
					global globalJogadores
					globalJogadores += 1
		self.pred = 0
		self.colisao = 0

	def atualiza_altura(self, x2, x1):
		altura = x2-x1
		if self.flagAlt:
			self.histAltura.append(altura)
			qtd_alt = len(self.histAltura)
			if qtd_alt == 10:
				self.flagAlt = False
			self.altura = math.ceil(sum(self.histAltura)/qtd_alt)
		else:
			del self.histAltura[9]
			self.histAltura.insert(0, altura)
			self.altura = math.ceil(sum(self.histAltura)/10)
		if (abs(self.x2 - x2) > 15) != (abs(self.x1 - x1) > 15):
			if abs(self.x2 - x2) > 15:
				self.x2 = self.x1 + (self.altura - 15)
			elif abs(self.x1 - x1) > 15:
				self.x1 = self.x2 - (self.altura - 15)
		else:
			self.x1 = x1
			self.x2 = x2
			self.altura = x2 - x1


	
	def quebra(self, colisor, time):
		#self.x1 = self.x2 - self.altura
		self.colisao += colisor.colisao
		if colisor.time != self.time and self.colisao >= 2:
			self.time = 0
			self.contTime = 0
			self.mediaTime = 0
			if time != 0:
				self.colisor = colisor


	def atualiza_vel(self, y2):
		if self.flagVel:
			self.histVel.append(y2 - self.y2)
			qtd_vel = len(self.histVel)
			if qtd_vel == 10:
				self.flagVel = False
			self.vel = math.ceil(sum(self.histVel)/qtd_vel)
		else:
			del self.histVel[9]
			self.histVel.insert(0, y2 - self.y2)
			self.vel = math.ceil(sum(self.histVel)/10)

	def ajusta(self, colisor):
		#self.atualiza_vel(y2)
		#self.atualiza_vel(y2)
		#self.x1 = x1
		self.colisao += 1
		if self.colisao >= 2:
			self.y1 = colisor.y1
			self.y2 = colisor.y2
			if colisor.time != self.time:
				self.time = 0
				self.contTime = 0
				self.mediaTime = 0
				if colisor.time != 0:
					self.colisor = colisor
		#else:
			#self.predict()
	

	def testa_borda(self):
		if (self.y2 < 40) or (self.y1 > w - 40):
			return True
		else:
			return False 

	def predict(self):
		# if flag:
		# 	self.x1 = self.x1+self.vel[0]
		# 	self.x2 = self.x2+self.vel[0]
		self.y1 = self.y1+self.vel
		self.y2 = self.y2+self.vel
		self.pred+=1
		#self.time = 2
		#return self.testa_borda()


	def getTuple1(self):
		return(self.y1, self.x1)

	def getTuple2(self):
		return(self.y2, self.x2)

class Goleira:
	def __init__(self):
		self.dist = [0, 0]
		self.x1 = [0, 0]
		self.x2 = [0, 0]
		self.y1 = [0, 0]
		self.y2 = [0, 0]
		self.mantem = False
		self.numTrave = 0

	def atualiza(self, x1, x2, y1, y2):
		self.mantem = True
		self.x1[self.numTrave] = x1
		self.x2[self.numTrave] = x2
		self.y1[self.numTrave] = y1
		self.y2[self.numTrave] = y2
		self.numTrave = (self.numTrave+1)%2

	def resetTraves(self):
		self.numTrave = 0
		self.mantem = False
		self.x1 = [0, 0]
		self.x2 = [0, 0]
		self.y1 = [0, 0]
		self.y2 = [0, 0]

	
	def getTuple1T1(self):
		return(self.y1[0], self.x1[0])

	def getTuple2T1(self):
		return(self.y2[0], self.x2[0])

	def getTuple1T2(self):
		return(self.y1[1], self.x1[1])

	def getTuple2T2(self):
		return(self.y2[1], self.x2[1])

class Bola:
	def __init__(self):
		self.dist = 0
		self.vel = [0, 0]
		self.mantem = False
		self.pred = 0
		self.x1 = 0
		self.x2 = 0
		self.y1 = 0
		self.y2 = 0
		self.dominio = False
		self.ok = 0
		self.predTotal = 0
		#self.detect = True
	
	def reset(self):
		self.dist = 0
		self.vel = [0, 0]
		self.mantem = False
		self.pred = 0
		self.x1 = 0
		self.x2 = 0
		self.y1 = 0
		self.y2 = 0
		self.dominio = False

	def atualiza(self, x1, x2, y1, y2):
		if self.x1:
			disx1, disx2, disy2, disy1 = abs(x1 - self.x1), abs(x2 - self.x2), abs(y2 - self.y2), abs(y1 - self.y1)
			if not (disx1 < 100 and disx2 < 100 and disy2 < 100 and disy1 < 100) and self.pred < 6:
				self.predict()
				return
		self.mantem = True
		self.vel = [x2 - self.x2, y2 - self.y2]
		# if self.detect:
		self.pred = 0
		#self.detec = True
		self.x1 = x1
		self.x2 = x2
		self.y1 = y1
		self.y2 = y2
		self.ok += 1

	def predict(self):
		self.predTotal += 1
		if self.dominio < 4:
			if self.pred == 0:
				self.lastPosition = [self.x2, self.y2]
			else:
				self.x1 += self.vel[0]
				self.x2 += self.vel[0]
				self.y1 += self.vel[1]
				self.y2 += self.vel[1] 
				if self.x1 < 0 or self.y1 < 0 or self.x2 > h or self.y2 > w:
					global erroBola
					erroBola += 1
		self.pred += 1
		self.mantem = True


	def getTuple1(self):
		return(self.y1, self.x1)

	def getTuple2(self):
		return(self.y2, self.x2)


class Objects:
	def __init__(self):
		self.listaJogadores = []
		self.listaJAux = []
		self.vago = 0
		self.bola = Bola()
		self.goleira = Goleira()
		self.CLUSTERS = 2
		self.listColors = []
		
		self.colorTimes = [np.array(3)]*2
		self.totalJogadores = 0

		#video 3
		#self.colorTimes = [[54, 49, 43], [165, 158, 134]]
		#video 1
		#self.colorTimes = [[79, 61, 46], [230, 220, 193]]
		self.contFrame = 0
		#self.colorTimes = [[191, 162, 138], [97, 49, 26]]


		self.ids = 0


		self.nReconhecidos = 0
		self.aglu = 0


	def kmeansColors(self, x1, x2, y1, y2, img):
		distY = y2 - y1
		distX = x2 - x1
		#tuple1 = (y1 + math.ceil(0/6 * distY), x1 + math.ceil(1/5 * distX))
		#tuple2 = (y1 + math.ceil(2/6 * distY), x1 + math.ceil(1/2 * distX))
		#tuple3 = (y1 + math.ceil(3/6 * distY), x1 + math.ceil(1/5 * distX))
		#tuple4 = (y1 + math.ceil(4/6 * distY), x1 + math.ceil(1/2 * distX))
		#tuple5 = (y1 + math.ceil(5/6 * distY), x1 + math.ceil(1/5 * distX))
		#tuple6 = (y1 + math.ceil(6/6 * distY), x1 + math.ceil(1/2 * distX))
		tuplet1 = (y1 + math.ceil(3/9 * distY), x1 + math.ceil(1/2 * distX))
		tuplet2 = (y1 + math.ceil(6/9 * distY), x1 + math.ceil(4/7 * distX))
		kmeans = KMeans(n_clusters = self.CLUSTERS)
		img2 = img[tuplet1[1]:tuplet2[1], tuplet1[0]:tuplet2[0]]
		img2 = img2.reshape((img2.shape[0] * img2.shape[1], 3))
		kmeans.fit(img2)
		COLORS = kmeans.cluster_centers_
		COLORS.astype(int)
		cv2.rectangle(img, tuplet1, tuplet2, amarelo, 2)
		#cv2.rectangle(img, tuple3, tuple4, np.ceil(COLORS[0]), 2)
		#cv2.rectangle(img, tuple4, tuple5, np.ceil(COLORS[1]), 2)
		#COLORS.sort(key = operator.itemgetter(0, 1, 2))
		return sorted(COLORS, key = operator.itemgetter(0, 1, 2))


	def update_list(self, COLORS):
		aux_best = -1
		for j, resultColor in enumerate(COLORS):
			best = -1
			aux = 80
			for i, color in enumerate(self.listColors):
				b = math.ceil(color[0][0]/color[1]) 
				g = math.ceil(color[0][1]/color[1])
				r = math.ceil(color[0][2]/color[1])  

				b, g, r  = abs(b - resultColor[0]), abs(g- resultColor[1]), abs(r - resultColor[2])
				if b < 40 and g < 40 and r < 40 and i != aux_best:
					result = b + g + r
					if aux > result:
						aux = result
						best = i

			if best != -1:
				b = math.ceil(self.listColors[best][0][0]/self.listColors[best][1]) 
				g = math.ceil(self.listColors[best][0][1]/self.listColors[best][1])
				r = math.ceil(self.listColors[best][0][2]/self.listColors[best][1]) 
				self.listColors[best][1] += 1
				self.listColors[best][0][0] += resultColor[0]
				self.listColors[best][0][1] += resultColor[1]
				self.listColors[best][0][2] += resultColor[2]
				aux_best = best
			else:
				self.listColors.append([resultColor, 1])

	def escolhe_time(self, COLORS):
		aux = 80
		valueT = 0
		testeBg = False
		#print("jogador:")
		for resultColor in   COLORS:
			b1 = abs(self.colorTimes[0][0] - resultColor[0])
			g1 = abs(self.colorTimes[0][1] - resultColor[1])
			r1 = abs(self.colorTimes[0][2] - resultColor[2])
			b2 = abs(self.colorTimes[1][0] - resultColor[0])
			g2 = abs(self.colorTimes[1][1] - resultColor[1])
			r2 = abs(self.colorTimes[1][2] - resultColor[2])


			if b1 < 30 and g1 < 30 and r1 < 30:
				result1 = b1 + g1 + r1
			else:
				result1 = 100
			if b2 < 30	and g2 < 30 and r2 < 30:
				result2 = b2 + g2 + r2
			else:
				result2 = 100


			if result2 < result1:
				if result2 < aux:
					aux = result2
					valueT = 0
					break
			elif result1 < result2:
				if result1 < aux:
					aux = result1
					valueT = 1

		#print(valueT)
		#print("\n\n\n")
		#if testeBg:
		return valueT
		#else:
		#	return -1




	def testa_time(self, x1, x2, y1, y2, img):
		COLORS = np.ceil(self.kmeansColors(x1, x2, y1, y2, img))
		if self.contFrame <= 20:
			self.update_list(COLORS)
			if self.contFrame == 20 or self.contFrame == 15:
				self.listColors.sort(key = lambda x: x[1], reverse=True)
				for color, qtd in self.listColors:
					print(np.ceil(color/qtd), qtd)
				#self.bg = np.ceil(self.listColors[0][0]/self.listColors[0][1])
				self.colorTimes[0] = np.ceil(self.listColors[0][0]/self.listColors[0][1])
				self.colorTimes[1] = np.ceil(self.listColors[1][0]/self.listColors[1][1])
		else:
			return self.escolhe_time(COLORS)	
		return -1


	def reset(self):
		self.listaJogadores = []
		self.bola.reset()
		self.goleira = Goleira()
		self.research = True

	
	def ajusta_objects(self):
		# for jogador in self.listaJAux:
		# 	print(jogador.x2)
		for jogador in self.listaJogadores:
			if not jogador.testa_borda():
				if jogador.cont >= 5:
					if jogador.pred < 10: # and jogador.colisao < 10:
						colisor = False
						#jogador.predict()
						for i, jogadorTest in enumerate(self.listaJAux):
							if  jogadorTest.x1 - jogador.x1 < 20:
								if jogador.x1 < jogadorTest.x2  and jogador.x2 > jogadorTest.x1 and jogador.y1 < jogadorTest.y2 and jogador.y2 > jogadorTest.y1:
									if not colisor or jogadorTest.x2 > colisor.x2:
										colisor = jogadorTest
										self.aglu += 1
						if colisor:
							time = jogador.time
							jogador.ajusta(colisor)
							colisor.quebra(jogador, time)
							colisor.predict()
						else:
							jogador.colisao = 0
							self.nReconhecidos += 1
						self.listaJAux.append(jogador)
						jogador.predict()
					else:
						self.nReconhecidos += 1

		self.listaJogadores = self.listaJAux
		self.listaJAux = []
		if not self.bola.mantem:
			self.bola.predict()
		# for jogador in self.listaJogadores:
		# 	print(jogador.x2)

	
	def draw_objects(self, img, frame_count):
		cv2.putText(img, str(frame_count), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
		self.contFrame+=1
		if self.listaJAux:		
			self.ajusta_objects()
			for jogador in self.listaJogadores:
				if jogador.time == 1:
					cv2.rectangle(img, jogador.getTuple1(), jogador.getTuple2(), vermelho, 2)
					cv2.putText(img, str(jogador.id), jogador.getTuple1(), font, fontScale/2, fontColor, lineType)
				elif jogador.time == 2:
					cv2.rectangle(img, jogador.getTuple1(), jogador.getTuple2(), azul, 2)
					cv2.putText(img, str(jogador.id), jogador.getTuple1(), font, fontScale/2, fontColor, lineType)
				else:
					cv2.rectangle(img, jogador.getTuple1(), jogador.getTuple2(), branco, 2)
					cv2.putText(img, str(jogador.id), jogador.getTuple1(), font, fontScale/2, fontColor, lineType)
			if self.bola.mantem:
				cv2.rectangle(img, self.bola.getTuple1(), self.bola.getTuple2(), amarelo, 2)
				self.bola.mantem = False
			if self.goleira.mantem:
				cv2.rectangle(img, self.goleira.getTuple1T1(), self.goleira.getTuple2T1(), preto, 2)
				if self.goleira.numTrave == 0:
					cv2.rectangle(img, self.goleira.getTuple1T2(), self.goleira.getTuple2T2(), preto, 2)
				self.goleira.mantem = False
			self.contJog = len(self.listaJogadores)

	def testa_borda(self, y1, y2):
		#print(y1, y2)
		if (y1 < 50) or (y2 > w - 50):
			return True
		else:
			return False



	def atualizaJogador(self, x1, x2, y1, y2, i, cont, img):
		jogador = self.listaJogadores[i-cont]
		if self.contFrame <= 20 or jogador.time or self.testa_borda(y1, y2):
			jogador.atualiza(x1, x2, y1, y2, -1, False) 
		else:
			time = self.testa_time(x1, x2, y1, y2, img)
			jogador.atualiza(x1, x2, y1, y2, time, False)
			self.totalJogadores += 1
		self.listaJAux.append(self.listaJogadores.pop(i-cont))



	def atualiza_list(self, lista_atualiza, img, research):
		#print(lista_atualiza)
		total = len(self.listaJogadores)
		#print(len(lista_atualiza), len(lista_classe))
		#lista_atualiza.sort(key = operator.itemgetter(1), reverse = True)
		#print(total)
		if research:
			self.reset()

		l_possiveis1 = [[[-1, 1000],  [-1, 1000]] for i in range(total)]
		l_possiveis2 = [[-1, 1000]  for i in range(len(lista_atualiza))]
		for j, detect in enumerate(lista_atualiza):
			x1, x2, y1, y2, classe = detect
			#cv2.imwrite('frames/img_j3_'+str(x1) +".png",img[x1: x2, y1:y2])
			#print(x1, x2, y1, y2)
			melhor = -1
			if classe == 1.:
				if x2 - x1 > 50:
					if self.contFrame <= 20:
						time = self.testa_time(x1, x2, y1, y2, img)
					flag = True
					for i, jogador in enumerate(self.listaJogadores):
						disx1, disx2, disy2, disy1 = abs(x1 - jogador.x1), abs(x2 - jogador.x2), abs(y2 - jogador.y2), abs(y1 - jogador.y1)
						#disP1 = abs(x1 - jogador.x1) + abs(y1 - jogador.y1)
						erro = 10 * jogador.colisao
						erroAlt = 10 * jogador.colisao
						if (disx1 < 50 or disx2 < 50 + erroAlt) and disy2 < 50 + erro and disy1 < 50 +erro :
							dist = 2*disx2+disy2+disx1+disy1
							if l_possiveis1[i][0][1] > dist:
								l_possiveis1[i][1] = l_possiveis1[i][0].copy()
								l_possiveis1[i][0] = [j, dist]
							elif l_possiveis1[i][1][1] > dist:
								l_possiveis1[i][1] = [j, dist]
							if l_possiveis2[j][1] > dist:
								l_possiveis2[j] = [i, dist]
							flag = False
					if flag:
						self.listaJAux.append(Jogador(x1, x2, y1, y2, -1, self.ids))
						self.totalJogadores += 1
						self.ids += 1
			elif classe == 2.:
				#if abs(x1 - self.bola.x1) < 20 or abs(x2 - self.bola.x2) < 20 or abs(y2 - self.bola.y2) < 20 or abs(y1 - self.bola.y1) < 20 or :
				self.bola.atualiza(x1, x2, y1, y2)
			else:
				if x2 - x1 > 50:
					self.goleira.atualiza(x1, x2, y1, y2)
		cont = 0
		for i, possiveis in enumerate(l_possiveis1):	
			testIndex, testDist = possiveis[0]
			pIndex, pDist = possiveis[1]
			if testIndex != -1 and i == l_possiveis2[testIndex][0]: 
				x1, x2, y1, y2, _ = lista_atualiza[testIndex]
				self.atualizaJogador(x1, x2, y1, y2, i, cont, img)
				cont+=1
			elif pIndex != -1 and testIndex == l_possiveis1[l_possiveis2[pIndex][0]][0][0]:
				x1, x2, y1, y2, _ = lista_atualiza[pIndex]
				self.atualizaJogador(x1, x2, y1, y2, i, cont, img)
				cont+=1
		# if total - len(self.listaJogadores) < 4:
		# 	return False
		# else:
		# 	self.ajusta_objects()
		# 	return True


class Event_detect(Objects):
	def __init__(self):
		#eventos
		Objects.__init__(self)
		self.posses = [0, 0]
		self.passesCE = [[0,0], [0,0]]
		self.passesInt = [0, 0]
		self.roubadas = [0, 0]
		self.dribles = [0, 0]
		
		self.jDominio = [None, 1]
		self.jDisputa = [None, 1]
		self.listaIndefinidos = []
		self.listPassesCorrige = []
		self.corrigeDisputaPR = []
		self.corrigeDisputaDR = []
		self.lastDominio = [0, None]
		self.lastDisputaID = -1
		self.lastEstado = 0
		self.listLancesA = []
		self.estado = 0
		self.flagDisputa = True
		# 0 = Bola livre
		# 1 = dominio
		# 2 = Disputa
		# 3 = Disputa Intensa
		self.totalDominios = 0
		#with open("checkPoint.txt", "r") as auxFile:
			#self.ids, self.posses[0], self.posses[1] = list(map(int, auxFile.read().split("\n")))



	def resetDetect(self, frame_count):
		self.jDominio = [None, 1]
		self.jDisputa = [None, 1]
		self.listaIndefinidos = []
		self.listPassesCorrige = []
		self.corrigeDisputaPR = []
		self.corrigeDisputaDR = []
		self.lastDominio = [0, None]
		self.lastDisputaID = -1
		self.lastEstado = 0
		self.estado = -1
		self.flagDisputa = True

		
	def detect(self, img, frame_count, research):
		if research:
			self.resetDetect(frame_count)
		aux = 60
		aux2 = 80
		jogadorDominio = None
		jogadorDisputa = None
		for i, jogador in enumerate(self.listaJogadores):
			distBolax2 =  abs(self.bola.x2 - jogador.x2)
			distBolay1 = distBolax2 + abs(self.bola.y1 - jogador.y1)
			distBolay2 = distBolax2 + abs(self.bola.y2 - jogador.y2)	
			if distBolay2 < distBolay1:
				if distBolay2 < aux:
					jogadorDisputa = jogadorDominio
					aux2 = aux
					jogadorDominio = jogador
					aux = distBolay2
				elif distBolay2 <= aux2:
					aux2 = distBolay2
					jogadorDisputa = jogador
			else:
				if distBolay1 < aux:
					jogadorDisputa = jogadorDominio
					aux2 = aux
					jogadorDominio = jogador
					aux = distBolay1
				elif distBolay1 <= aux2:
					aux2 = distBolay1
					jogadorDisputa = jogador
			if jogadorDisputa and not jogadorDominio:
				jogadorDisputa = None
			elif jogadorDisputa and jogadorDominio and jogadorDisputa.time == jogadorDominio.time:
				jogadorDisputa = None

		self.defineEstado(jogadorDominio, jogadorDisputa, frame_count)
		self.defineEvento(frame_count)
		self.bolaBaseJogador()
		self.draw_Dominios(img)
		#self.printResult(frame_count)

 

	def defineEstado(self, jogadorDominio, jogadorDisputa, frame_count):
		if self.jDominio[0] and jogadorDominio:
			if jogadorDominio.id == self.jDominio[0].id:
				self.bola.dominio += 1
				self.jDominio[1] += 1
				if self.jDominio[1] >= 6:
					self.estado = 1
			else:
				if self.jDisputa[0] and jogadorDisputa:
					if self.jDisputa[0].id == jogadorDominio.id and self.jDominio[0].id == jogadorDisputa.id:
						aux = self.jDominio.copy()
						self.jDominio = self.jDisputa
						self.jDisputa = aux
						self.jDisputa[1] += 1
						self.jDominio[1] += 1
						self.estado = 2
				else:
					if self.jDominio[1] >= 5:
						self.jDisputa = self.jDominio.copy()
						self.estado = 2
					else:
						self.jDisputa = self.jDominio.copy()
						self.estado = 0
					self.jDominio[0] = jogadorDominio
					self.jDominio[1] = 1
				self.bola.dominio = 1
		elif jogadorDominio:
			if self.lastDominio[1] and jogadorDominio.id == self.lastDominio[1].id:
				self.jDominio[1] = 3
			else:
				self.jDominio[1] = 1
			self.jDominio[0] = jogadorDominio
			self.ajustaJogadorDisputa(jogadorDisputa)
			self.bola.dominio = 1
		elif self.jDominio[0]:
			self.jDominio = [None, 1]
			self.bola.dominio = 0
			self.estado = 0
		else:
			self.jDominio[1] += 1	
			self.estado = 0
		
		if self.estado != 2:
			self.ajustaJogadorDisputa(jogadorDisputa)

		if self.jDominio[0] and self.jDisputa[0]:
			if self.jDominio[0].colisao >= 2:
				self.listLancesA.append(frame_count)


	def defineEvento(self, frame_count):
		if self.jDominio[0]:
			print(self.lastDominio[0], self.jDominio[0].time, self.estado, self.lastEstado)
		else:
			print(self.lastDominio[0], -1, self.estado, self.lastEstado)

		self.corrigeLances()
		self.checkLastDominio()
		if self.estado == 1:
			if self.lastEstado == 0:
				if self.lastDominio[1]:
					self.checkPasse(frame_count)
			elif self.lastEstado == 5:
				if self.lastDominio[1] and self.jDominio[0].id != self.lastDominio[1].id:
					self.checkRoubada(frame_count)
			elif self.lastEstado == 2:
				if self.lastDominio[1]:
					if self.jDominio[0].id ==  self.lastDominio[1].id or self.jDominio[0].id  == self.lastDisputaID:
						self.checkDribleRoubada(frame_count)
					else:
						self.checkPasseRoubada(frame_count)
			elif self.lastEstado == 4:
				if self.aux < 3:
					self.checkPasse(frame_count)
				else:
					self.checkPasseRoubada(frame_count)
			if self.lastDominio != self.jDominio:
				self.totalDominios += 1
			if self.lastDisputaID:
				self.lastDisputaID = 0
			self.checkPosse()
			self.lastDominio = [self.jDominio[0].time, self.jDominio[0]] 
		elif self.estado == 2:
			if self.lastEstado == 0 or self.lastEstado == 5:
				self.estado = 4
				self.aux = 1
			elif self.lastEstado == 4:
				self.estado = 4
				self.aux += 1
			if self.jDisputa[0] and not self.lastDisputaID:
				self.lastDisputaID = self.jDisputa[0].id
		elif self.estado == 0:
			if self.lastEstado == 1:
				self.lastEstado = self.estado
				return
			else:
				return
		self.lastEstado = self.estado
		
	def ajustaJogadorDisputa(self, jogadorDisputa):
		if self.jDisputa[0] and jogadorDisputa:
			if self.jDisputa[0].id == jogadorDisputa.id:
				self.jDisputa[1] += 1
				if self.jDisputa[1] >= 2:
					self.estado = 2
			else:
				self.jDisputa[0] = jogadorDisputa
				self.jDisputa[1] = 1
				self.flagDisputa = True
		elif jogadorDisputa:
			self.jDisputa[0] = jogadorDisputa
			self.jDisputa[1] = 1
		elif self.jDisputa[0] and self.flagDisputa:
			self.flagDisputa = False
			if self.jDisputa[1] >= 5:
				self.estado = 2
		else:
			self.jDisputa = [None, 1]



	def corrigeLances(self):
		self.corrigePasseRoubada()
		self.corrigeDribleRoubada()
		self.corrigePasse()
		self.corrigePosse()

	def checkRoubada(self, frame_count):
		if self.lastDominio[0] != 0 and self.jDominio[0].time != 0:
			if self.lastDominio[0] == self.jDominio[0].time:
				print(str(frame_count) + ": Passe certo")
				self.passesCE[self.jDominio[0].time - 1][0] += 1
			elif self.lastDominio[0] != self.jDominio[0].time:
				print(str(frame_count) + ": Roubada")
				self.roubadas[self.jDominio[0].time - 1] += 1
		elif self.lastDominio[0] != 0:
			self.corrigeDisputaDR.append([self.lastDominio[0], self.jDominio[0], frame_count])


	def corrigePasseRoubada(self):
		aux = []
		for time1, jogador, frame_count in self.corrigeDisputaPR:
			if jogador.time != 0:
				if time1 == jogador.time:
					self.passesCE[time1 - 1][0] += 1
					self.dribles[time1 - 1] += 1
					print(str(frame_count) + ": Passe certo e drible [Correcao]")
				elif time1 !=  jogador.time:
					self.roubadas[jogador.time - 1] += 1
					self.passesCE[jogador.time - 1][0] += 1
					print(str(frame_count) + ": Roubada e passe certo [Correcao]")
			else:
				aux.append([time1, jogador, frame_count])
		self.corrigeDisputaPR = aux

	def checkPasseRoubada(self, frame_count):
		if self.lastDominio[0] != 0 and self.jDominio[0].time != 0:
			if self.lastDominio[0] == self.jDominio[0].time:
				self.passesCE[self.jDominio[0].time - 1][0] += 1
				self.dribles[self.jDominio[0].time - 1] += 1
				print(str(frame_count) + ": Passe certo e individual")
			elif self.lastDominio[0] != self.jDominio[0].time:
				self.roubadas[self.jDominio[0].time - 1] += 1
				self.passesCE[self.jDominio[0].time - 1][0] += 1
				print(str(frame_count) + ": Roubada e passe certo")
		elif self.lastDominio[0] != 0:
			self.corrigeDisputaPR.append([self.lastDominio[0], self.jDominio[0], frame_count])

	def corrigeDribleRoubada(self):
		aux = []
		for time1, jogador, frame_count in self.corrigeDisputaDR:
			if jogador.time != 0:
				print(jogador.time, time1)
				if time1 == jogador.time:
					self.dribles[time1 - 1] += 1
					print(str(frame_count) + ": Drible [Correcao]")
				elif time1 !=  jogador.time:
					self.roubadas[jogador.time - 1] += 1
					print(str(frame_count) + ": Roubada [Correcao] ")
			else:
				aux.append([time1, jogador, frame_count])
		self.corrigeDisputaDR = aux

	def checkDribleRoubada(self, frame_count):
		if self.lastDominio[0] != 0 and self.jDominio[0].time != 0:
			if self.lastDominio[0] == self.jDominio[0].time:
				self.dribles[self.jDominio[0].time - 1] += 1
				print(str(frame_count) + ": Drible")
			elif self.lastDominio[0] != self.jDominio[0].time:
				self.roubadas[self.jDominio[0].time - 1] += 1
				print(str(frame_count) + ": Roubada")
		elif self.lastDominio[0] != 0:
			self.corrigeDisputaDR.append([self.lastDominio[0], self.jDominio[0], frame_count])

	def checkLastDominio(self):
		if not self.lastDominio[0]:
			if self.lastDominio[1] and  self.lastDominio[1].time:
				self.lastDominio[0] = self.lastDominio[1].time

	def corrigePasse(self):
		aux = []
		for jogador, lastJogador, frame_count in self.listPassesCorrige:
			if jogador and lastJogador and jogador.time and lastJogador.time:
				if jogador.time == lastJogador.time:
					self.passesCE[lastJogador.time - 1][0] += 1
					print(str(frame_count) + ": Passe certo [Correcao]")
				else:
					self.passesCE[lastJogador.time - 1][1] += 1	
					self.passesInt[lastJogador.time %2] += 1
					print(str(frame_count) + ": Passse errado [Correcao]")
			else:
				aux.append([jogador, lastJogador, frame_count])
		self.listPassesCorrige = aux

	def checkPasse(self, frame_count):
		if self.lastDominio[0] != 0 and self.jDominio[0].time != 0:
			if self.lastDominio[1].id != self.jDominio[0].id:
				if self.lastDominio[0] == self.jDominio[0].time:
					self.passesCE[self.lastDominio[0] - 1][0] += 1
					print(str(frame_count) + ": Passe certo")
				else:
					self.passesCE[self.lastDominio[0] - 1][1] += 1
					self.passesInt[self.lastDominio[0] %2] += 1
					print(str(frame_count) + ": Passse errado")
			else:
				self.posses[self.jDominio[0].time - 1] += 2
		else:
			self.listPassesCorrige.append([self.jDominio[0], self.lastDominio[1], frame_count])

	
	def corrigePosse(self):
		aux = []
		for jogador, qtd in self.listaIndefinidos:
			if jogador.time != 0:
				self.posses[jogador.time - 1] += qtd
			else:
				aux.append([jogador, qtd])
		self.listaIndefinidos = aux


	def checkPosse(self):
		if self.jDominio[0].time != 0:
			self.posses[self.jDominio[0	 ].time - 1] += 1
		else:
			self.somaDominioIndef()





	def draw_Dominios(self, img):
		if self.jDominio[0]:
			cv2.rectangle(img, self.jDominio[0].getTuple1(), self.jDominio[0].getTuple2(), preto, 2)
		if self.jDisputa[0]:
			cv2.rectangle(img, self.jDisputa[0].getTuple1(), self.jDisputa[0].getTuple2(), amarelo, 2)





	def printResult(self, frame_count):
		totalPosse = sum(self.posses)
		if totalPosse:
			print ("Time 1:")
			print ("Passes Certos: " + str(self.passesCE[0][0]))
			print ("Passes Errados: " + str(self.passesCE[0][1]))
			print ("Posse de bola: " + str(self.posses[0]/totalPosse*100))
			print ("Interceptacoes: " + str(self.passesInt[0]))
			print ("Roubo de bola: " + str (self.roubadas[0]))
			print ("Lance Individual: " + str(self.dribles[0]))
			print ("Time 2:")
			print ("Passes Certos: " + str(self.passesCE[1][0]))
			print ("Passes Errados: " + str(self.passesCE[1][1]))
			print ("Posse de bola: " + str(self.posses[1]/totalPosse*100))
			print ("Interceptacoes: " + str(self.passesInt[1]))
			print ("Roubo de bola: " + str (self.roubadas[1]))
			print ("Lance Individual: " + str(self.dribles[1]))
			print ("-----------------------------")
			print ("Lances para serem avaliados: ")
			for frame_count in self.listLancesA:
				print(str(frame_count))





	def bolaBaseJogador(self):

		if self.jDominio[0] and self.bola.pred > 3:
			self.bola.x2 = self.jDominio[0].x2
			self.bola.x1 = self.jDominio[0].x2 - math.ceil(1/5*(self.jDominio[0].altura))
			self.bola.y1 = self.jDominio[0].y1 + math.ceil(2/5*(self.jDominio[0].y2 - self.jDominio[0].y1))
			self.bola.y2 = self.jDominio[0].y1 + math.ceil(4/5*(self.jDominio[0].y2 - self.jDominio[0].y1))
			self.bola.domino = True

	def somaDominioIndef(self):
		for i, jogador in enumerate(self.listaIndefinidos):
			if self.jDominio[0].id == jogador[0].id:
				self.listaIndefinidos[i][1] += 1
				return
		self.listaIndefinidos.append([self.jDominio[0], 1])









					
					