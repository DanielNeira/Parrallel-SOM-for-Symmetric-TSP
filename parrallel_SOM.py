import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib as mpl
from intersection import doIntersect
from scipy.spatial import ConvexHull
from multiprocessing import Process, Manager
from main import som_from_outside,som_from_outside_2
from opt import three_opt
import time

def read_tsp(filename):
    """
    Read a file in .tsp format into a pandas DataFrame

    The .tsp files can be found in the TSPLIB project. Currently, the library
    only considers the possibility of a 2D map.
    """
    with open('instancias/'+filename) as f:
        node_coord_start = None
        dimension = None
        lines = f.readlines()

        # Obtiene la información del .tsp
        i = 0
        while not dimension or not node_coord_start:
            line = lines[i]
            if line.startswith('DIMENSION :'):
                dimension = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i
            i = i+1

        print('Instacia '+filename+' con {} ciudades leído.'.format(dimension))

        f.seek(0)

        # Leer un data frame en base al tsp
        cities = pd.read_csv(
            f,
            skiprows=node_coord_start + 1,
            sep=' ',
            names=['city', 'y', 'x'],
            dtype={'city': str, 'x': np.float64, 'y': np.float64},
            header=None,
            nrows=dimension
        )

        # cities.set_index('city', inplace=True)
        # para pasarlo a lista de 1-D numpy array
        cities_list = list()
        for index, row in cities.iterrows():
            cities_list.append(np.asarray(row[['x','y']]))
        #print(cities_list)
        return cities_list,cities
    
def mean_shift(problem):
    bandwidth = estimate_bandwidth(problem, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(problem)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    return labels, cluster_centers

def k_means(problem):
    kmeans = KMeans(n_clusters=20, random_state=12340)
    kmeans.fit(problem)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return labels, cluster_centers

def are_they_nb(i,j):
    for k in range(n_clusters_):
        for l in range(n_clusters_):
            if (k > l) and (i != k and j != k) and (i != l and j != l):
                if  (doIntersect(cluster_centers[i],cluster_centers[j],
                                    cluster_centers[k],cluster_centers[l])):
                    print("adyacentes[",i,",",j,"]: False, por interseccion")
                    #print("k,l: ",k,l)
                    return False
                else:
                    #print("k,l: ",k,l)
                    prom = 0
                    for index in range(n_clusters_):
                        prom += np.linalg.norm(cluster_centers[i]-cluster_centers[index])
                    prom = prom/(n_clusters_) #podrias calcular esto antes y solo llamarlo
                    if np.linalg.norm(cluster_centers[i]-cluster_centers[j]) >= prom:
                        #print("||P_a P_b||",np.linalg.norm(cluster_centers[i]-cluster_centers[j]),"E_r||P_a P_r||",prom)
                        print("adyacentes[",i,",",j,"]: False, por distancia")
                        return False
    return True  

def convex_hull(return_list_hull,cluster,points):
    hull = ConvexHull(points)
    return_list_hull[cluster] = hull

def add_cluster_centers(a,b,cluster_centers):
    """Adiciona el cluster_center del nuevo cluster creado"""
    equis_a = cluster_centers[a][0]
    eye_a = cluster_centers[a][1]
    equis_b = cluster_centers[b][0]
    eye_b = cluster_centers[b][1]
    ananais = np.array([(equis_a*len(return_list_per[a])+equis_b*len(return_list_per[b]))
                  /(len(return_list_per[a])+len(return_list_per[b])),
                  (eye_a*len(return_list_per[a])+eye_b*len(return_list_per[b]))
                  /(len(return_list_per[a])+len(return_list_per[b]))])
    return np.concatenate((cluster_centers, [ananais]), axis=0)

instancia = 'ei8246' #'qa194','uy734','ar9152','fi10639','it16862','ei8246'
problem,cities = read_tsp(instancia+'.tsp')

start = time.clock()

labels, cluster_centers = k_means(cities[['x', 'y']])

print("Tiempo después de clusterizar: %f " %(time.clock()-start))

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Número de Clusters estimados: %d" % n_clusters_)
cities["cluster"] = labels

#Hacer los SOMs en multithread
manager = Manager()
return_list_per = manager.list(range(n_clusters_))
return_list_dist = manager.list(range(n_clusters_))
return_distancia = manager.list(range(n_clusters_))
threads = list()

for k in range(n_clusters_):
    t = Process(target=som_from_outside_2, args=(cities[cities["cluster"] == k], 
                                                return_list_per,return_list_dist,k,return_distancia))
    threads.append(t)
    t.start()

for proc in threads:
    proc.join()

#Aca voy a poner el convex hull
manager = Manager()
return_list_hull = manager.list(range(n_clusters_))
threads = list()
for k in range(n_clusters_):
    t = Process(target=convex_hull,args=(return_list_hull,k,cities[cities["cluster"] == k][['x','y']].as_matrix()))
    threads.append(t)
    t.start()

for proc in threads:
    proc.join()

clusters_activos = [i for i in range(n_clusters_)]

conecciones = []
breakfor = False
for i in range(n_clusters_): 
    #print("-----------------Análisis nodo %d-----------------" %(i))
    prom = 0
    for index in range(n_clusters_):
        prom += np.linalg.norm(cluster_centers[i]-cluster_centers[index])
    prom = prom/(n_clusters_ - 1)
    for j in range(n_clusters_):
        if i < j:
            #print("----Analizando si nodo %d es vecino---" %j)
            if np.linalg.norm(cluster_centers[i]-cluster_centers[j]) < prom:
                for k in range(n_clusters_):
                    if k != i and k != j: 
                        #print("--Viendo si cluster %d está estorbando" %k)
                        if np.linalg.norm(cluster_centers[i]-cluster_centers[k]) < prom:
                            points = cities[cities["cluster"] == k][['x','y']].as_matrix()
                            vertices = return_list_hull[k].vertices
                            #print('vertices',vertices)
                            #print('vertices[:-1]',vertices[:-1])
                            for l in range(len(vertices[:-1])):
                                #print(type(points[vertices[l]]),points[vertices[i]])
                                #print(type(cluster_centers[i]),cluster_centers[i])
                                if(doIntersect(cluster_centers[i],cluster_centers[j],
                                                        points[vertices[l]],points[vertices[l+1]])):
                                    #print("Hay interseccion entre centroides %d y %d" %(i,j))
                                    breakfor = True
                                    break
                            if breakfor:
                                break
                            if(doIntersect(cluster_centers[i],cluster_centers[j],
                                                        points[vertices[-1]],points[vertices[0]])):
                                print("Hay interseccion entre centroides %d y %d" %(i,j))
                                breakfor = True
                                break
                            #print("Cluster %d no estorba" %k)
                        else:
                            #print("No se considera nodo %d porque está más lejos del promedio" %k)
                            pass
                if breakfor:
                    breakfor = False
                    continue
                #print("Nodos %d y %d son vecinos, wii" %(i,j))
                conecciones.append((i,j))
            else:
                pass
                #print("Más lejano al promedio")
print('conecciones',conecciones) #aca tenemos a los vecinos de los weones
#print("-------------------acaaaa------------------1")
while len(conecciones) != 0: #correr la anterior
    (a,b) = conecciones[0] #Se saca el primer par de la lista (se va a ir eliminando todos los pares que contengan a uno del par escogido)
    print("(a,b)=(%d,%d)"%(a,b))
    eliminar1 = [i for i, (c,d) in enumerate(conecciones) if c == a]
    [conecciones.pop(i) for i in eliminar1[::-1]]
    eliminar2 = [i for i, (c,d) in enumerate(conecciones) if d == b]
    [conecciones.pop(i) for i in eliminar2[::-1]]
    eliminar3 = [i for i, (c,d) in enumerate(conecciones) if c == b]
    [conecciones.pop(i) for i in eliminar3[::-1]]
    points_a = cities[cities["cluster"] == a][['x','y']].as_matrix()
    #print("points_a:",points_a)
    vertices_a = return_list_hull[a].vertices
    points_b = cities[cities["cluster"] == b][['x','y']].as_matrix()
    vertices_b = return_list_hull[b].vertices
    mejores = {} #aca se calcula cual de los puntos dentro del par de convexhull son los mas cercanos
    mejor_resultado_hasta_ahora = 999999999999
    for xy_a,ver_a in zip(points_a[vertices_a],vertices_a):
        for xy_b,ver_b in zip(points_b[vertices_b],vertices_b):
            resultado = np.linalg.norm(xy_a-xy_b)
            if (resultado < mejor_resultado_hasta_ahora):
                mejores[resultado] = [(xy_a,xy_b),(ver_a,ver_b)]
    mejormejormejor = list(mejores.keys())[0]            
    mejores = [value for (key, value) in sorted(mejores.items())]
    [(elec_a , elec_b),(elec_xy_a , elec_xy_b)] = mejores.pop(0)
    ciudad_a = cities[cities.cluster == a].iloc[elec_xy_a]['city']
    ciudad_b = cities[cities.cluster == b].iloc[elec_xy_b]['city']
    return_list_per[a] = return_list_per[a].reset_index(drop=True)
    anterior_a = return_list_per[a].iloc[return_list_per[a].loc[return_list_per[a].city == ciudad_a].index-1]
    anterior_a = anterior_a[['x','y']].values
    if len(return_list_per[a]) == return_list_per[a].loc[return_list_per[a].city == ciudad_a].index+1:
        posterior_a = return_list_per[a].loc[0]
        posterior_a = posterior_a[['x','y']].values
    else:
        posterior_a = return_list_per[a].iloc[return_list_per[a].loc[return_list_per[a].city == ciudad_a].index+1]
        posterior_a = posterior_a[['x','y']].values
    return_list_per[b] = return_list_per[b].reset_index(drop=True)
    anterior_b = return_list_per[b].iloc[return_list_per[b].loc[return_list_per[b].city == ciudad_b].index-1]
    anterior_b = anterior_b[['x','y']].values
    if len(return_list_per[b]) == return_list_per[b].loc[return_list_per[b].city == ciudad_b].index+1:
        posterior_b = return_list_per[b].loc[0]
        posterior_b = posterior_b[['x','y']].values
    else:
        posterior_b = return_list_per[b].iloc[return_list_per[b].loc[return_list_per[b].city == ciudad_b].index+1]
        posterior_b = posterior_b[['x','y']].values
    ##Ahora unimos los clusters usando los datos calculados
    if np.linalg.norm(posterior_a-anterior_b) < np.linalg.norm(anterior_a-posterior_b):
        distancia_menor = np.linalg.norm(posterior_a-anterior_b)
        salorcito_a = return_list_per[a].loc[return_list_per[a].city == ciudad_a].index
        salorcito_b = return_list_per[b].loc[return_list_per[b].city == ciudad_b].index
        mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]+1])
        mamalosita = mamalosita.append(return_list_per[b].iloc[salorcito_b[0]:], ignore_index=True)
        distancia_lado_derecho, distancia_lado_izquierdo = -1, -1
        if salorcito_b[0] != 0:
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]], ignore_index=True)
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]-1][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]][['x','y']].values )
        if salorcito_a[0]+1 != len(return_list_per[a]):
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]+1:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]+1][['x','y']].values )
        return_list_per.append(mamalosita)
        if distancia_lado_derecho == -1:
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[-1][['x','y']].values )
        if distancia_lado_izquierdo ==  -1:
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[0][['x','y']].values )

        return_distancia.append(return_distancia[a] + return_distancia[b] - distancia_lado_derecho - distancia_lado_izquierdo
                                          + mejormejormejor + distancia_menor)
        cluster_centers = add_cluster_centers(a,b,cluster_centers)
        clusters_activos.append(n_clusters_)
        n_clusters_ += 1
    else:
        distancia_menor = np.linalg.norm(anterior_a-posterior_b)
        #caso predecesor a sucesor b
        salorcito_a = return_list_per[a].loc[return_list_per[a].city == ciudad_a].index
        salorcito_b = return_list_per[b].loc[return_list_per[b].city == ciudad_b].index
        if salorcito_a[0] != 0 and salorcito_b[0] + 1 != len(return_list_per[b]):
            mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]])
            mamalosita = mamalosita.append(return_list_per[b].iloc[salorcito_b[0]+1:], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]-1][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]+1][['x','y']].values )
        elif salorcito_b[0] + 1 != len(return_list_per[b]):
            mamalosita = pd.DataFrame.copy(return_list_per[b].iloc[salorcito_b[0]+1:])
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[-1][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]+1][['x','y']].values )
        elif salorcito_a[0] != 0:
            mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]])
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]-1][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[0][['x','y']].values )
        else:
            mamalosita = pd.DataFrame.copy(return_list_per[b].iloc[:salorcito_b[0]+1])
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[-1][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[0][['x','y']].values )
        return_list_per.append(mamalosita)
        return_distancia.append(return_distancia[a] + return_distancia[b] - distancia_lado_derecho - distancia_lado_izquierdo
                                                 + mejormejormejor + distancia_menor)
        cluster_centers = add_cluster_centers(a,b,cluster_centers)
        clusters_activos.append(n_clusters_)
        n_clusters_ += 1
    clusters_activos.remove(a)
    clusters_activos.remove(b)
print("clusters_activos 0",clusters_activos)

while(len(clusters_activos) > 1):
    n_clusters_a_agregar = 0
    clusters_a_agregar = []
    for num in clusters_activos:
        if len(return_list_hull)-1 < num:
            n_clusters_a_agregar += 1
            clusters_a_agregar.append(num)
 
    for i in range(n_clusters_a_agregar):
        return_list_hull.append(None)
    #append Convex hull a los conjuntos creados
    threads = list()
    for k in clusters_a_agregar:
        t = Process(target=convex_hull,args=(return_list_hull,k,return_list_per[k]))
        threads.append(t)
        t.start()
    for proc in threads:
        proc.join()
    conecciones = [] 
    breakfor = False
    for i in clusters_activos: 
        #print("-----------------Análisis nodo %d-----------------" %(i))
        prom = 0
        for index in clusters_activos:
            prom += np.linalg.norm(cluster_centers[i]-cluster_centers[index])
        prom = prom/(len(clusters_activos) - 1)
        for j in clusters_activos:
            if i < j:
                #print("----Analizando si nodo %d es vecino---" %j)
                if np.linalg.norm(cluster_centers[i]-cluster_centers[j]) < prom:
                    for k in clusters_activos:
                        if k != i and k != j: 
                            #print("--Viendo si cluster %d está estorbando" %k)
                            if np.linalg.norm(cluster_centers[i]-cluster_centers[k]) < prom:
                                points = return_list_per[k][['x','y']].as_matrix()
                                vertices = return_list_hull[k].vertices
                                #print('vertices',vertices)
                                #print('vertices[:-1]',vertices[:-1])
                                for l in range(len(vertices[:-1])):
                                    #print(type(points[vertices[l]]),points[vertices[i]])
                                    #print(type(cluster_centers[i]),cluster_centers[i])
                                    if(doIntersect(cluster_centers[i],cluster_centers[j],
                                                            points[vertices[l]],points[vertices[l+1]])):
                                        #print("Hay interseccion entre centroides %d y %d" %(i,j))
                                        breakfor = True
                                        break
                                if breakfor:
                                    break
                                if(doIntersect(cluster_centers[i],cluster_centers[j],
                                                            points[vertices[-1]],points[vertices[0]])):
                                    print("Hay interseccion entre centroides %d y %d" %(i,j))
                                    breakfor = True
                                    break
                                #print("Cluster %d no estorba" %k)
                            else:
                                #print("No se considera nodo %d porque está más lejos del promedio" %k)
                                pass
                    if breakfor:
                        breakfor = False
                        continue
                    #print("Nodos %d y %d son vecinos, wii" %(i,j))
                    conecciones.append((i,j))
                else:
                    pass
                    #print("Más lejano al promedio")
    print('conecciones 1',conecciones) #aca tenemos a los vecinos de los weones    
    print("clusters_activos sd",clusters_activos)

    if conecciones == []:
        break

    while  len(conecciones) != 0: #correr la anterior
        (a,b) = conecciones[0] #Se saca el primer par de la lista (se va a ir eliminando todos los pares que contengan a uno del par escogido)
        print("(a,b)=(%d,%d)"%(a,b))
        eliminar1 = [i for i, (c,d) in enumerate(conecciones) if c == a]
        [conecciones.pop(i) for i in eliminar1[::-1]]
        eliminar2 = [i for i, (c,d) in enumerate(conecciones) if d == b]
        [conecciones.pop(i) for i in eliminar2[::-1]]
        eliminar3 = [i for i, (c,d) in enumerate(conecciones) if c == b]
        [conecciones.pop(i) for i in eliminar3[::-1]]
        points_a = return_list_per[a][['x','y']].as_matrix()
        vertices_a = return_list_hull[a].vertices
        points_b = return_list_per[b][['x','y']].as_matrix()
        vertices_b = return_list_hull[b].vertices
        mejores = {} #aca se calcula cual de los puntos dentro del par de convexhull son los mas cercanos
        mejor_resultado_hasta_ahora = 999999999999
        for xy_a,ver_a in zip(points_a[vertices_a],vertices_a):
            for xy_b,ver_b in zip(points_b[vertices_b],vertices_b):
                resultado = np.linalg.norm(xy_a-xy_b)
                if (resultado < mejor_resultado_hasta_ahora):
                    mejores[resultado] = [(xy_a,xy_b),(ver_a,ver_b)]
        mejormejormejor = list(mejores.keys())[0]         
        mejores = [value for (key, value) in sorted(mejores.items())]
        [(elec_a , elec_b),(elec_xy_a , elec_xy_b)] = mejores.pop(0)
        ciudad_a = return_list_per[a].iloc[elec_xy_a]['city']
        ciudad_b = return_list_per[b].iloc[elec_xy_b]['city']
        return_list_per[a] = return_list_per[a].reset_index(drop=True)
        anterior_a = return_list_per[a].iloc[return_list_per[a].loc[return_list_per[a].city == ciudad_a].index-1]
        anterior_a = anterior_a[['x','y']].values
        if len(return_list_per[a]) == return_list_per[a].loc[return_list_per[a].city == ciudad_a].index+1:
            posterior_a = return_list_per[a].loc[0]
            posterior_a = posterior_a[['x','y']].values
        else:
            posterior_a = return_list_per[a].iloc[return_list_per[a].loc[return_list_per[a].city == ciudad_a].index+1]
            posterior_a = posterior_a[['x','y']].values
        return_list_per[b] = return_list_per[b].reset_index(drop=True)
        anterior_b = return_list_per[b].iloc[return_list_per[b].loc[return_list_per[b].city == ciudad_b].index-1]
        anterior_b = anterior_b[['x','y']].values
        if len(return_list_per[b]) == return_list_per[b].loc[return_list_per[b].city == ciudad_b].index+1:
            posterior_b = return_list_per[b].loc[0]
            posterior_b = posterior_b[['x','y']].values
        else:
            posterior_b = return_list_per[b].iloc[return_list_per[b].loc[return_list_per[b].city == ciudad_b].index+1]
            posterior_b = posterior_b[['x','y']].values
        if np.linalg.norm(posterior_a-anterior_b) > np.linalg.norm(anterior_a-posterior_b):
            distancia_menor = np.linalg.norm(posterior_a-anterior_b)
            salorcito_a = return_list_per[a].loc[return_list_per[a].city == ciudad_a].index
            salorcito_b = return_list_per[b].loc[return_list_per[b].city == ciudad_b].index
            mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]+1])
            mamalosita = mamalosita.append(return_list_per[b].iloc[salorcito_b[0]:], ignore_index=True)
            distancia_lado_derecho, distancia_lado_izquierdo = -1, -1
            if salorcito_b[0] != 0:
                mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]], ignore_index=True)
                distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]-1][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]][['x','y']].values )
            if salorcito_a[0]+1 != len(return_list_per[a]):
                mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]+1:], ignore_index=True)
                distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]+1][['x','y']].values )
            return_list_per.append(mamalosita)
            if distancia_lado_derecho == -1:
                distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[-1][['x','y']].values )
            if distancia_lado_izquierdo ==  -1:
                distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                       return_list_per[a].iloc[0][['x','y']].values )

            return_distancia.append(return_distancia[a] + return_distancia[b] - distancia_lado_derecho - distancia_lado_izquierdo
                                          + mejormejormejor + distancia_menor)
            cluster_centers = add_cluster_centers(a,b,cluster_centers)
            clusters_activos.append(n_clusters_)
            n_clusters_ += 1
        else:
            distancia_menor = np.linalg.norm(anterior_a-posterior_b)
            #caso predecesor a sucesor b
            salorcito_a = return_list_per[a].loc[return_list_per[a].city == ciudad_a].index
            salorcito_b = return_list_per[b].loc[return_list_per[b].city == ciudad_b].index
            if salorcito_a[0] != 0 and salorcito_b[0] + 1 != len(return_list_per[b]):
                mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]])
                mamalosita = mamalosita.append(return_list_per[b].iloc[salorcito_b[0]+1:], ignore_index=True)
                mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
                mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
                distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]-1][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]][['x','y']].values )
                distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]+1][['x','y']].values )
            elif salorcito_b[0] + 1 != len(return_list_per[b]):
                mamalosita = pd.DataFrame.copy(return_list_per[b].iloc[salorcito_b[0]+1:])
                mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
                mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
                distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[-1][['x','y']].values )
                distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]+1][['x','y']].values )
            elif salorcito_a[0] != 0:
                mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]])
                mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
                mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
                distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]-1][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]][['x','y']].values )
                distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[0][['x','y']].values )
            else:
                mamalosita = pd.DataFrame.copy(return_list_per[b].iloc[:salorcito_b[0]+1])
                mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
                distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[-1][['x','y']].values )
                distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[0][['x','y']].values )
            return_list_per.append(mamalosita)
            return_distancia.append(return_distancia[a] + return_distancia[b] - distancia_lado_derecho - distancia_lado_izquierdo
                                                 + mejormejormejor + distancia_menor)
            cluster_centers = add_cluster_centers(a,b,cluster_centers)
            clusters_activos.append(n_clusters_)
            n_clusters_ += 1
        clusters_activos.remove(a)
        clusters_activos.remove(b)
print("clusters_activos 2",clusters_activos)

if len(clusters_activos) == 2:
    (a,b) = clusters_activos 
    print("(a,b)=(%d,%d)"%(a,b))
    points_a = return_list_per[a][['x','y']].as_matrix()
    #print("points_a:",points_a)
    vertices_a = return_list_hull[a].vertices
    points_b = return_list_per[b][['x','y']].as_matrix()
    vertices_b = return_list_hull[b].vertices
    mejores = {} #aca se calcula cual de los puntos dentro del par de convexhull son los mas cercanos
    mejor_resultado_hasta_ahora = 999999999999
    for xy_a,ver_a in zip(points_a[vertices_a],vertices_a):
        for xy_b,ver_b in zip(points_b[vertices_b],vertices_b):
            resultado = np.linalg.norm(xy_a-xy_b)
            if (resultado < mejor_resultado_hasta_ahora):
                mejores[resultado] = [(xy_a,xy_b),(ver_a,ver_b)]
    mejormejormejor = list(mejores.keys())[0]   
    mejores = [value for (key, value) in sorted(mejores.items())]
    [(elec_a , elec_b),(elec_xy_a , elec_xy_b)] = mejores.pop(0)
    ciudad_a = return_list_per[a].iloc[elec_xy_a]['city']
    ciudad_b = return_list_per[b].iloc[elec_xy_b]['city']
    return_list_per[a] = return_list_per[a].reset_index(drop=True)
    anterior_a = return_list_per[a].iloc[return_list_per[a].loc[return_list_per[a].city == ciudad_a].index-1]
    anterior_a = anterior_a[['x','y']].values
    if len(return_list_per[a]) == return_list_per[a].loc[return_list_per[a].city == ciudad_a].index+1:
        posterior_a = return_list_per[a].loc[0]
        posterior_a = posterior_a[['x','y']].values
    else:
        posterior_a = return_list_per[a].iloc[return_list_per[a].loc[return_list_per[a].city == ciudad_a].index+1]
        posterior_a = posterior_a[['x','y']].values
    return_list_per[b] = return_list_per[b].reset_index(drop=True)
    anterior_b = return_list_per[b].iloc[return_list_per[b].loc[return_list_per[b].city == ciudad_b].index-1]
    anterior_b = anterior_b[['x','y']].values
    if len(return_list_per[b]) == return_list_per[b].loc[return_list_per[b].city == ciudad_b].index+1:
        posterior_b = return_list_per[b].loc[0]
        posterior_b = posterior_b[['x','y']].values
    else:
        posterior_b = return_list_per[b].iloc[return_list_per[b].loc[return_list_per[b].city == ciudad_b].index+1]
        posterior_b = posterior_b[['x','y']].values
    if np.linalg.norm(posterior_a-anterior_b) < np.linalg.norm(anterior_a-posterior_b):
        distancia_menor = np.linalg.norm(posterior_a-anterior_b)
        salorcito_a = return_list_per[a].loc[return_list_per[a].city == ciudad_a].index
        salorcito_b = return_list_per[b].loc[return_list_per[b].city == ciudad_b].index
        mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]+1])
        mamalosita = mamalosita.append(return_list_per[b].iloc[salorcito_b[0]:], ignore_index=True)
        distancia_lado_derecho, distancia_lado_izquierdo = -1, -1
        if salorcito_b[0] != 0:
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]], ignore_index=True)
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]-1][['x','y']].values -
                                                   return_list_per[b].iloc[salorcito_b[0]][['x','y']].values )
        if salorcito_a[0]+1 != len(return_list_per[a]):
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]+1:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]+1][['x','y']].values )
        return_list_per.append(mamalosita)
        if distancia_lado_derecho == -1:
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[-1][['x','y']].values )
        if distancia_lado_izquierdo ==  -1:
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[0][['x','y']].values )
        return_distancia.append(return_distancia[a] + return_distancia[b] - distancia_lado_derecho - distancia_lado_izquierdo
                                         + mejormejormejor + distancia_menor)
        cluster_centers = add_cluster_centers(a,b,cluster_centers)
        clusters_activos.append(n_clusters_)
        n_clusters_ += 1
    else:
        distancia_menor = np.linalg.norm(anterior_a-posterior_b)
        #caso predecesor a sucesor b
        salorcito_a = return_list_per[a].loc[return_list_per[a].city == ciudad_a].index
        salorcito_b = return_list_per[b].loc[return_list_per[b].city == ciudad_b].index
        if salorcito_a[0] != 0 and salorcito_b[0] + 1 != len(return_list_per[b]):
            mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]])
            mamalosita = mamalosita.append(return_list_per[b].iloc[salorcito_b[0]+1:], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]-1][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]+1][['x','y']].values )
        elif salorcito_b[0] + 1 != len(return_list_per[b]):
            mamalosita = pd.DataFrame.copy(return_list_per[b].iloc[salorcito_b[0]+1:])
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[-1][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[salorcito_b[0]+1][['x','y']].values )
        elif salorcito_a[0] != 0:
            mamalosita = pd.DataFrame.copy(return_list_per[a].iloc[:salorcito_a[0]])
            mamalosita = mamalosita.append(return_list_per[b].iloc[:salorcito_b[0]+1], ignore_index=True)
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]-1][['x','y']].values -
                                                    return_list_per[a].iloc[salorcito_a[0]][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[0][['x','y']].values )
        else:
            mamalosita = pd.DataFrame.copy(return_list_per[b].iloc[:salorcito_b[0]+1])
            mamalosita = mamalosita.append(return_list_per[a].iloc[salorcito_a[0]:], ignore_index=True)
            distancia_lado_izquierdo = np.linalg.norm( return_list_per[a].iloc[salorcito_a[0]][['x','y']].values -
                                                    return_list_per[a].iloc[-1][['x','y']].values )
            distancia_lado_derecho = np.linalg.norm(return_list_per[b].iloc[salorcito_b[0]][['x','y']].values -
                                                    return_list_per[b].iloc[0][['x','y']].values )
        return_list_per.append(mamalosita)
        return_distancia.append(return_distancia[a] + return_distancia[b] - distancia_lado_derecho - distancia_lado_izquierdo
                                         + mejormejormejor + distancia_menor)
        cluster_centers = add_cluster_centers(a,b,cluster_centers)
        clusters_activos.append(n_clusters_)
        n_clusters_ += 1
    clusters_activos.remove(a)
    clusters_activos.remove(b)

print("Distancia final = %f, tiempo= %f" %(return_distancia[-1], time.clock() - start ))

#return_list_per[-1],return_distancia[-1] = three_opt(return_list_per[-1],return_distancia[-1], 1)

#print("Después de 3-opt, Distancia final = %f, tiempo= %f"%(return_distancia[-1], time.clock() - start ))
