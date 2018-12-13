import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib as mpl
from intersection import doIntersect
from scipy.spatial import ConvexHull

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

instancia = 'ar9152'
problem,cities = read_tsp(instancia+'.tsp')
print(type(cities),cities.head())
#print(type(problem),problem[123][0], len(problem))

def mean_shift(problem):
    bandwidth = estimate_bandwidth(problem, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(problem)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    return labels, cluster_centers

def k_means(problem):
    kmeans = KMeans(n_clusters=23, random_state=12340)
    kmeans.fit(problem)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return labels, cluster_centers

#labels, cluster_centers = mean_shift(problem)
#labels, cluster_centers = k_means(problem)
labels, cluster_centers = k_means(cities[['x', 'y']])
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Número de Clusters estimados: %d" % n_clusters_)
#print(cities.head(10))
cities["cluster"] = labels
#print(cities.head(10))
# #############################################################################

# Plot resultados, algo tiene que no me gusto
if False:
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = (labels == k)
        #print(my_members)
        cluster_center = cluster_centers[k]
        for i in range(len(problem)):
        	if my_members[i]:
        		plt.plot(problem[i][0], problem[i][1],col+ '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Número de Clusters estimados: %d' % n_clusters_)
    plt.show()
    """
    #for k in range(n_clusters_):
    #    print(cities[cities["cluster"]== k].head(5))
    """
    #print(problem)
    subproblems = []
    subproblem = []
    for k in range(n_clusters_):
        my_members = (labels == k)
        for i in range(len(problem)):
            if my_members[i]:
                #print("Cluster ",k,problem[i][0], problem[i][1])
                subproblem.append([problem[i][0], problem[i][1]])
                #subproblem.append(np.asarray(problem[i][0], problem[i][1]))
        subproblems.append(subproblem)
        subproblem = []
    #print(subproblems[0],"\n\n\n",subproblems[k])

#plotea los clusters intento 2 -> creo que el mejor
if False:
    fig, ax = plt.subplots()
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure(figsize=(5, 5), frameon = False)
    axis = fig.add_axes([0,0,1,1])
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k,col in zip(range(n_clusters_),colors):
        my_members = (labels == k)
        cluster_center = cluster_centers[k]
        for i in range(len(problem)):
            if my_members[i]:
                plt.plot(problem[i][0], problem[i][1],col+'.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        ax.annotate(str(k),xy=(cluster_center[0], cluster_center[1]), color='white')
    plt.title('Número de Clusters estimados: %d' % n_clusters_)
    #plt.show()
    plt.savefig('centroides.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    

if False: #mostrar los centroides
    fig, ax = plt.subplots()
    for k in range(n_clusters_):
        cluster_center = cluster_centers[k]
        ax.annotate(str(k),xy=(cluster_center[0], cluster_center[1]))
    plt.title('Número de Clusters estimados: %d' % n_clusters_)
    ax.set_ylim(0.9*np.min(cluster_centers[:,1]), 1.1*np.max(cluster_centers[:,1]))
    ax.set_xlim(0.9*np.min(cluster_centers[:,0]), 1.1*np.max(cluster_centers[:,0]))
    plt.show()

#Hacer los SOMs en multithread
if True:
    if True:
        from multiprocessing import Process, Manager
        from main import som_from_outside,som_from_outside_2
        from itertools import cycle


        #som_from_outside_2(cities)
        manager = Manager()
        return_list_per = manager.list(range(n_clusters_))
        return_list_dist = manager.list(range(n_clusters_))
        threads = list()
        for k in range(n_clusters_):
        #for k in range(2):
            t = Process(target=som_from_outside_2, args=(cities[cities["cluster"] == k], 
                                                        return_list_per,return_list_dist,k))
            threads.append(t)
            t.start()

        for proc in threads:
            proc.join()

        #print("return_list_dist de cluster 0\n",return_list_dist[0])
        print("return_list_per de 0 \n",return_list_per[0])
        print("return_list_per de 0 \n",return_list_per[10])

    #plot_route(cities, route, 'pruebita')

    #plotea la solucion inicial de los SOMs para unir 
    if False:
        mpl.rcParams['agg.path.chunksize'] = 10000
        fig = plt.figure(figsize=(5, 5), frameon = False)
        axis = fig.add_axes([0,0,1,1])
        colors = cycle('bgcmykbgcmykbgcmykbgcmyk')
        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')
        #axis.scatter(cities['x'], cities['y'], color='red', s=4)
        #axis.plot(return_list_per[0]['x'], return_list_per[0]['y'] , linewidth=1)
        #axis.plot(return_list_per[1]['x'], return_list_per[1]['y'], linewidth=1)
        for i, col in zip(range(n_clusters_), colors):
        #for i, col in zip([0,10], colors):
            axis.plot(return_list_per[i]['x'], return_list_per[i]['y'], col, linewidth=1)
            if False:
                points = cities[cities["cluster"] == i][['x','y']].as_matrix()
                axis.scatter(points[:,0], points[:,1], color='red', s=1)
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        #plt.savefig('pruebita_1', bbox_inches='tight', pad_inches=0, dpi=200)
        #plt.close()
        plt.show()

#calcula los adyacentes -> no sirve mucho
if False:
    adyacentes = np.ones((n_clusters_,n_clusters_), dtype=bool) #all set to True
    #print(adyacentes)
    salto = False
    for i in range(n_clusters_):
        for j in range(n_clusters_):
    #for i in range(3):
    #    for j in range(3):
            if (i > j):
                for k in range(n_clusters_):
                    for l in range(n_clusters_):
                        if (k > l) and (i != k and j != k) and (i != l and j != l):
                            if  (doIntersect(cluster_centers[i],cluster_centers[j],
                                                cluster_centers[k],cluster_centers[l])):
                                adyacentes[i,j] = False
                                print("adyacentes[",i,",",j,"]: False, por interseccion")
                                print("k,l: ",k,l)
                                salto = True
                                break
                            else:
                                print("k,l: ",k,l)
                                prom = 0
                                for index in range(n_clusters_):
                                    prom += np.linalg.norm(cluster_centers[i]-cluster_centers[index])
                                prom = prom/(n_clusters_) #podrias calcular esto antes y solo llamarlo
                                if np.linalg.norm(cluster_centers[i]-cluster_centers[j]) >= prom:
                                    adyacentes[i,j] = False
                                    print("||P_a P_b||",np.linalg.norm(cluster_centers[i]-cluster_centers[j]),"E_r||P_a P_r||",prom)
                                    print("adyacentes[",i,",",j,"]: False, por distancia")
                                    salto = True
                                    break
                    if salto:
                        salto = False
                        break
                    print("prueba 1:",i,j,k,l)
                #if salto:
                #    salto = False
                #    break
                #print("prueba 2:",i,j,k,l)
            #if salto:
            #    salto = False
            #    break
            #print("prueba 3:",i,j)

    for i in range(n_clusters_):
        for j in range(n_clusters_):
    #for i in range(3):
    #    for j in range(3):
            if (i > j):
                if adyacentes[i,j]:
                    print("Son vecinos: (i=",i," , j=",j,")")

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
#para verificar
if False:
    for i in range(n_clusters_):
        for j in range(n_clusters_):
            if i != j:
                if are_they_nb(i,j)==True:
                    print(i,j,"son vecinos")

if False: #plotea pero haciendo CONVEXHULL en un core
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig = plt.figure(figsize=(5, 5), frameon = False)
    axis = fig.add_axes([0,0,1,1])
    colors = cycle('bgcmykbgcmykbgcmykbgcmyk')
    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    #axis.scatter(cities['x'], cities['y'], color='red', s=2)
    for k in range(n_clusters_):
        points = cities[cities["cluster"] == k][['x','y']].as_matrix()
        #print("Aca viene los puntos con cluster igual a 1")
        #print(points.head(10))
        #print(points)
        hull = ConvexHull(points)
        plt.plot(points[:,0], points[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
        #plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
        plt.annotate(str(k),xy=(cluster_centers[k][0], cluster_centers[k][1]), color='red')
    #plt.show()
    plt.savefig('todo.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

#Aca voy a poner el convex hull
if True:
    if True:
        if False:
            def convex_hull(return_list_hull,return_list_orden,cluster,points):
                hull = ConvexHull(points)
                return_list_hull.append(hull)
                return_list_orden.append(cluster)
                

            from multiprocessing import Process, Manager
            
            manager = Manager()
            return_list_hull = manager.list(range(n_clusters_))
            #return_list_orden = manager.list()
            threads = list()
            for k in range(n_clusters_):
                t = Process(target=convex_hull, args=(return_list_hull,return_list_orden,k,
                            cities[cities["cluster"] == k][['x','y']].as_matrix()))
                threads.append(t)
                t.start()

            for proc in threads:
                proc.join()


            #print(return_list_hull[0]) <--- no te la devuelve necesariamente en orden!!
            #print(type(len(cities[cities["cluster"] == 1])))
            #for k in range(len(return_list_orden)):
            #    print("Cluster %d de largo supuesto %d, largo real %d" %(return_list_orden[k],len(return_list_hull[k].points),
            #        len(cities[cities["cluster"] == k])))

            #ploter al tiro la cosa
            if False:
                mpl.rcParams['agg.path.chunksize'] = 10000
                fig = plt.figure(figsize=(5, 5), frameon = False)
                axis = fig.add_axes([0,0,1,1])
                colors = cycle('bgcmykbgcmykbgcmykbgcmyk')
                axis.set_aspect('equal', adjustable='datalim')
                plt.axis('off')
                for k, col in zip(return_list_orden,colors):
                    points = cities[cities["cluster"] == k][['x','y']].as_matrix()
                    axis.scatter(points[:,0], points[:,1], color = col, s=2)
                    #print(points[return_list_hull[k].vertices,0])
                    #plt.plot(points[return_list_hull[k].vertices,0], points[return_list_hull[k].vertices,1], 'r--', lw=2)
                    for simplex in return_list_hull[k].simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], 'r--', lw=2 )
                    #print(points[return_list_hull[k].vertices,0], points[return_list_hull[k].vertices[0],1])
                    #plt.plot(points[return_list_hull[k].vertices[-1],0], points[return_list_hull[k].vertices[0],1], 'r--',lw=12)
                    print('bien hasta %d' %(k))
                #plt.show()
        if True:
            def convex_hull(return_list_hull,cluster,points):
                hull = ConvexHull(points)
                return_list_hull[cluster] = hull
                
            from multiprocessing import Process, Manager
            
            manager = Manager()
            return_list_hull = manager.list(range(n_clusters_))
            threads = list()
            for k in range(n_clusters_):
                t = Process(target=convex_hull,args=(return_list_hull,k,cities[cities["cluster"] == k][['x','y']].as_matrix()))
                threads.append(t)
                t.start()

            for proc in threads:
                proc.join()

            for k in range(n_clusters_):
                print("Cluster %d de largo supuesto %d, largo real %d" %(k,len(return_list_hull[k].points),
                    len(cities[cities["cluster"] == k])))
            
            #ploter al tiro la cosa
            if False:
                mpl.rcParams['agg.path.chunksize'] = 10000
                fig = plt.figure(figsize=(5, 5), frameon = False)
                axis = fig.add_axes([0,0,1,1])
                colors = cycle('bgcmykbgcmykbgcmykbgcmyk')
                axis.set_aspect('equal', adjustable='datalim')
                plt.axis('off')
                for k, col in zip(range(n_clusters_),colors):
                    points = cities[cities["cluster"] == k][['x','y']].as_matrix()
                    axis.scatter(points[:,0], points[:,1], color = col, s=2)
                    #print(points[return_list_hull[k].vertices,0])
                    #plt.plot(points[return_list_hull[k].vertices,0], points[return_list_hull[k].vertices,1], 'r--', lw=2)
                    for simplex in return_list_hull[k].simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], 'r--', lw=2 )
                    #print(points[return_list_hull[k].vertices,0], points[return_list_hull[k].vertices[0],1])
                    #plt.plot(points[return_list_hull[k].vertices[-1],0], points[return_list_hull[k].vertices[0],1], 'r--',lw=12)
                    print('bien hasta %d' %(k))
                plt.show()
    if False:
        return_list_hull = []
        for k in range(n_clusters_):
            points = cities[cities["cluster"] == k][['x','y']].as_matrix()
            return_list_hull.append(ConvexHull(points))
    #print(return_list_hull[0].vertices)
    #print(cities[cities["cluster"] == 0][['x','y']].as_matrix()[return_list_hull[0].vertices[0]])
    #ahora vamos con los cercanos
    if True:
        conecciones = []
        breakfor = False
        for i in range(n_clusters_): 
            #print("-----------------Análisis nodo %d-----------------" %(i))
            prom = 0
            for index in range(n_clusters_):
                prom += np.linalg.norm(cluster_centers[i]-cluster_centers[index])
            prom = prom/(n_clusters_)
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
        while  len(conecciones) != 0:
            (a,b) = conecciones[0] #Se saca el primer par de la lista (se va a ir eliminando todos los pares que contengan a uno del par escogido)
            #print("\n",a,b,"\n")
            eliminar1 = [i for i, (c,d) in enumerate(conecciones) if c == a]
            [conecciones.pop(i) for i in eliminar1[::-1]]
            eliminar2 = [i for i, (c,d) in enumerate(conecciones) if d == b]
            [conecciones.pop(i) for i in eliminar2[::-1]]
            eliminar3 = [i for i, (c,d) in enumerate(conecciones) if c == b]
            [conecciones.pop(i) for i in eliminar3[::-1]]
            #print(conecciones)
            #print("ConvexHull de cluster %d:" %a, return_list_hull[a].vertices)
            #lowest1 = 1000000000
            #pos1 = [-1,-1]
            #lowest2 = 1000000000
            #pos2 = [-1,-1]
            points_a = cities[cities["cluster"] == a][['x','y']].as_matrix()
            print("points_a:",points_a)
            vertices_a = return_list_hull[a].vertices
            points_b = cities[cities["cluster"] == b][['x','y']].as_matrix()
            vertices_b = return_list_hull[b].vertices
            #for k, ver_a in enumerate(vertices_a):
            #    for l, ver_b in enumerate(vertices_b):
            #        resultado = np.linalg.norm(ver_a-ver_b)
            """
                    if resultado < lowest1:
                        lowest1 = resultado
                        if pos2[0] != pos1[0] and pos2[0] != pos1[1] and pos2[1] != pos1[0] and pos2[1] != pos1[1]:
                            pos2 = pos1
                        pos1 = [k,l]
                        if pos2[0] == pos1[0] or pos2[0] == pos1[1] or pos2[1] == pos1[0] or pos2[1] == pos1[1]:
                            pos2 = [-1,-1]
                    elif resultado < lowest2 and k != pos1[0] and k != pos1[1] and l != pos1[0] and l != pos1[1]:
                        lowest2 = resultado
                        pos2 = [k,l]
            """
            mejores = {} #aca se calcula cual de los puntos dentro del par de convexhull son los mas cercanos
            mejor_resultado_hasta_ahora = 999999999999
            print("vertices_a:\n",vertices_a)
            print("vertices_b:\n",vertices_b)
            for xy_a,ver_a in zip(points_a[vertices_a],vertices_a):
                for xy_b,ver_b in zip(points_b[vertices_b],vertices_b):
                    resultado = np.linalg.norm(xy_a-xy_b)
                    if (resultado < mejor_resultado_hasta_ahora):
                        mejores[resultado] = [(xy_a,xy_b),(ver_a,ver_b)]
            mejores = [value for (key, value) in sorted(mejores.items())]
            print("mejores:",mejores)
            print("mejores[o]:",mejores[0])
            [(elec_a , elec_b),(elec_xy_a , elec_xy_b)] = mejores.pop(0)
            print("elec_a:",elec_a,"\nelec_b:", elec_b,"\nelec_xy_a:",elec_xy_a,"\nelec_xy_b:",elec_xy_b)
            #exit()
            #ploter para ver si se está seleccionando los puntos correctos y que wea
            if False:
                mpl.rcParams['agg.path.chunksize'] = 10000
                fig = plt.figure(figsize=(5, 5), frameon = False)
                axis = fig.add_axes([0,0,1,1])
                axis.set_aspect('equal', adjustable='datalim')
                plt.axis('off')
                axis.scatter(points_a[:,0], points_a[:,1], color = 'red', s=2)
                axis.scatter(points_b[:,0], points_b[:,1], color = 'blue', s=2)
                for k in [a,b]:
                    points = cities[cities["cluster"] == k][['x','y']].as_matrix()
                    for simplex in return_list_hull[k].simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], 'g--', lw=2 )
                #plt.plot(points[return_list_hull[k].vertices,0], points[return_list_hull[k].vertices,1], 'r--', lw=2)
                plt.plot([elec_a[0],elec_b[0]], [elec_a[1],elec_b[1]], 'r--', lw=2)
                #print(points[return_list_hull[k].vertices,0], points[return_list_hull[k].vertices[0],1])
                #plt.plot(points[return_list_hull[k].vertices[-1],0], points[return_list_hull[k].vertices[0],1], 'r--',lw=12)
                plt.show()
            #if doIntersect(points_a[g],points_b[h],points_a[t],points_b[w]): #cambiar porq esto son puntos necesito las cordenadas
            #    temp = h
            #    h = w
            #    w = temp
            #print("valor de a: %d" %(a))
            #print("ciudad\n", cities[cities["cluster"]==a])
            #print("----------------------------------------")
            #print("ciudad parece\n", cities[cities["city"]==elec_xy_a+1])
            #print("----------------------------------------")
            #print("ciudad parece SIN +1\n", cities[cities["city"]==elec_xy_a])
            #print("----------------------------------------")
            #print("ciudad\n", cities[cities["cluster"]==a].iloc[elec_xy_a])
            #print("----------------------------------------")
            #print("ciudad\n", cities.iloc[elec_xy_a])
            #dos = cities.iloc[elec_xy_a]['city']
            #print("dos",dos)
            ciudad_a = cities.iloc[elec_xy_a]['city']
            print("Imprimes esto para una idea return_list_per[a]\n",return_list_per[a])
            print("orwdasd \n", return_list_per[a].get_loc(ciudad_a))
            exit()
            location_in_route_a = return_list_per[a][return_list_per[a]['city'] == ciudad_a].loc()
            print("location_in_route_a:\n",location_in_route_a)
            anterior_a = return_list_per[a][return_list_per[a]== location_in_route_a - 1][['x','y']]
            posterior_a = return_list_per[a][return_list_per[a]== location_in_route_a + 1][['x','y']]
            ciudad_b = cities.iloc[elec_xy_b]['city']
            location_in_route_b = return_list_per[b][return_list_per[b] == ciudad_b].loc()
            anterior_b = return_list_per[a][return_list_per[a]== location_in_route_a - 1][['x','y']]
            posterior_b = return_list_per[a][return_list_per[a]== location_in_route_a + 1][['x','y']]
            exit()
            distancia_menor = 10000000
            puntos1_elegido = [-1,-1]
            puntos2_elegido = [-1,-1]
            if np.linalg.norm(anterior_a-posterior_a) < distancia_menor: #por ahora no analizaré si hay kinks
                distancia_menor = np.linalg.norm(anterior_a-posterior_a)
                puntos1_elegido = anterior_a
                puntos2_elegido = posterior_a
            elif np.linalg.norm(anterior_b-posterior_b) < distancia_menor: 
                distancia_menor = np.linalg.norm(anterior_b-posterior_b)
                puntos1_elegido = anterior_b
                puntos2_elegido = posterior_b
            elif np.linalg.norm(anterior_a-anterior_b) < distancia_menor:# and intersection(anterior_a,anterior_b,ver_a,ver_b)
                distancia_menor = np.linalg.norm(anterior_a-anterior_b)
                puntos1_elegido = anterior_a
                puntos2_elegido = anterior_b
            elif np.linalg.norm(posterior_a-posterior_b) < distancia_menor:
                distancia_menor = np.linalg.norm(posterior_a-posterior_b)
                puntos1_elegido = posterior_a
                puntos2_elegido = posterior_b
            exit()

def three_opt(p, broad=False):
    """In the broad sense, 3-opt means choosing any three edges ab, cd
    and ef and chopping them, and then reconnecting (such that the
    result is still a complete tour). There are eight ways of doing
    it. One is the identity, 3 are 2-opt moves (because either ab, cd,
    or ef is reconnected), and 4 are 3-opt moves (in the narrower
    sense)."""
    n = len(p)
    # choose 3 unique edges defined by their first node
    a, c, e = random.sample(range(n+1), 3)
    # without loss of generality, sort
    a, c, e = sorted([a, c, e])
    b, d, f = a+1, c+1, e+1

    if broad == True:
        which = random.randint(0, 7) # allow any of the 8
    else:
        which = random.choice([3, 4, 5, 6]) # allow only strict 3-opt

    # in the following slices, the nodes abcdef are referred to by
    # name. x:y:-1 means step backwards. anything like c+1 or d-1
    # refers to c or d, but to include the item itself, we use the +1
    # or -1 in the slice
    if which == 0:
        sol = p[:a+1] + p[b:c+1]    + p[d:e+1]    + p[f:] # identity
    elif which == 1:
        sol = p[:a+1] + p[b:c+1]    + p[e:d-1:-1] + p[f:] # 2-opt
    elif which == 2:
        sol = p[:a+1] + p[c:b-1:-1] + p[d:e+1]    + p[f:] # 2-opt
    elif which == 3:
        sol = p[:a+1] + p[c:b-1:-1] + p[e:d-1:-1] + p[f:] # 3-opt
    elif which == 4:
        sol = p[:a+1] + p[d:e+1]    + p[b:c+1]    + p[f:] # 3-opt
    elif which == 5:
        sol = p[:a+1] + p[d:e+1]    + p[c:b-1:-1] + p[f:] # 3-opt
    elif which == 6:
        sol = p[:a+1] + p[e:d-1:-1] + p[b:c+1]    + p[f:] # 3-opt
    elif which == 7:
        sol = p[:a+1] + p[e:d-1:-1] + p[c:b-1:-1] + p[f:] # 2-opt

    return sol


