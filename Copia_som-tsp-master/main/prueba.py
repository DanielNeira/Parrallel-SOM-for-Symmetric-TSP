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
size = 20
network = np.zeros((size,2))
for i in range(size):
	for j in range(2):
		if j == 0:
			network[i,j] = np.cos(i*2*np.pi/float(size))
		else:
			network[i,j] = np.sin(i*2*np.pi/float(size))

print(network)

#result_array = np.empty((0, 100))
#data_array = np.zeros((0,100))
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
#print(result_array)