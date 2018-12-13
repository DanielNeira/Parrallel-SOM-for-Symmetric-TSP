import numpy as np
import pandas as pd
#x= [12,321,12312,3213,123,321]
#x = np.array(x)
#print(x.shape[0])
#print(10//3)
#print(x[:,np.newaxis])
"""
df = pd.DataFrame({'x':[1,2,3,4,455,32,6,6,7,32],
					'y':[12,342,63,44,45,2,62,61,2,3],
					'city':[1,2,3,4,5,6,7,8,9,10]})

DF =df[['x','y']].values.tolist()

print(df)
print(DF)
print(type(DF),type(DF[0]))
cities=list()
for index, row in df.iterrows():
	#ondi=row[['x','y']]
	#cities.append(np.asarray(ondi))
	cities.append(np.asarray(row[['x','y']]))
print(cities)

vector_entrada1 =list([np.asarray([565.0, 575.0]),np.asarray([25.0, 185.0]),np.asarray([345.0, 750.0]),
				np.asarray([945.0, 685.0]),np.asarray([845.0, 655.0])])
print(vector_entrada1)
"""
"""
size = 20
network = np.zeros((size,2))
for i in range(size):
	for j in range(2):
		if j == 0:
			network[i,j] = np.cos(i*2*np.pi/float(size))
		else:
			network[i,j] = np.sin(i*2*np.pi/float(size))

print(network)
"""
#result_array = np.empty((0, 100))
#data_array = np.zeros((0,100))
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
#print(result_array)
"""
n_clusters_ = 10
prom = 20
ite = [245,5,3345,4,5,3345,5,4,453,4]
ite2 = [434,5,5,5,4,3]
breakfor = False
coneccion = []
for i in range(n_clusters_-1): 
	print("----------------------------Análisis nodo %d-----------------" %(i))
	for j in range(n_clusters_):
		if i < j:
			print("------------Analizando si nodo %d es vecino-------" %j)
			if ite[j] < prom:
				for k in range(n_clusters_):
					if k != i and k != j: 
						print("----Viendo si cluster %d está estorbando---" %k)
						if ite[k] < prom:
							for l in range(3):
								if ite2[l] > prom:
									print("x-Hay interseccion entre centroides %d y %d-x" %(i,j))
									breakfor = True
									break
							if breakfor:
								print("Break 7")
								#breakfor = False
								break
							if ite2[5] > prom:
								print("x-Hay interseccion entre centroides %d y %d en el 5-x" %(i,j))
								breakfor = True
								break
							print("-Cluster %d no estorba" %k)
						else:
							print("--No se considera nodo %d porque está más lejos del promedio" %k)
						#if breakfor:
						#	print("Break 1")
						#	break
					#if breakfor:
					#	print("Break 2")
					#	breakfor = False
					#	break
				if breakfor:
					print("Break 3")
					breakfor = False
					continue
				print("Nodos %d y %d son vecinos, wii" %(i,j))
				coneccion.append((i,j))
			else:
				print("--Nodo %d Más lejano al promedio" %j)
		if breakfor:
			print("Break 15")
		#	breakfor = False
		#	break
print(coneccion)
"""
"""
coneccion = [(0, 3), (0, 7), (0, 10), (0, 17), (0, 18), (0, 22), (1, 8), (1, 14), (1, 20), (2, 9), (2, 15), (2, 16), (3, 7), (3, 13), (3, 18), (3, 21), (4, 18), 
			(4, 21), (5, 8), (5, 11), (5, 12), (6, 19), (7, 13), (7, 15), (7, 22), (8, 11), (8, 16), (8, 20), (9, 11), (9, 15), (9, 16), (9, 17), (9, 22), (10, 12), 
			(10, 17), (10, 21), (11, 12), (11, 17), (12, 17), (13, 15), (14, 19), (14, 20), (15, 22), (16, 20), (17, 22), (18, 21)]
print(coneccion)
while  len(coneccion) != 0:
	(a,b) = coneccion[0]
	print("\n",a,b,"\n")
	eliminar1 = [i for i, (c,d) in enumerate(coneccion) if c == a]
	[coneccion.pop(i) for i in eliminar1[::-1]]
	eliminar2 = [i for i, (c,d) in enumerate(coneccion) if d == b]
	[coneccion.pop(i) for i in eliminar2[::-1]]
	eliminar3 = [i for i, (c,d) in enumerate(coneccion) if c == b]
	[coneccion.pop(i) for i in eliminar3[::-1]]
	print(coneccion)
"""
holo = {5:(5,6),10:(5,7)}

holo[34] = (8,10)
holo[24] = (1,2)
holo[1] = (2,6)
holo[2] = (2,4)
print(holo)
holo2 = sorted(holo, key=holo.__getitem__)
print(holo2)
holo3 = [value for (key, value) in sorted(holo.items())]
print(holo3)

a,b = holo3[0]
print("El más corto ",holo3[0])
for c,d in holo3:
	if c != a and c != b and d != a and d != b:
		print("El segundo más corto",(c,d))
		break






