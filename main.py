import numpy as np
import time
from io_helper import read_tsp, normalize,normalize_2
from neuron import generate_network, get_neighborhood, get_route, generate_circular_network_center
from distance import select_closest, euclidean_distance, route_distance
from plot import plot_network, plot_route


def write(instancia, num_max_iter, delta_learning_rate, delta_n, distance, learning_rate, factor_neuronas, corrida,
        tiempo_entrenamiento, tiempo_resolucion,tiempo_total, generation):
    """
    Esta función toma los strings motor, solucion, x_sol, tiemp_ejecutado, y si plot es True grafica la sol
    primera columna: instancia, segunda: función objetivo y tercera: tiemp_ejecucion
    """ 
    if generation == 1:
        metodo_iniciacion = 'Random_cuadrado'
    elif generation ==2:
        metodo_iniciacion = 'Circular'
    archivo = open('resultados.txt','a')
    archivo.write(str(instancia)+';'+str(metodo_iniciacion)+';'+str(num_max_iter)+';'+str(delta_learning_rate)+';'+str(delta_n)+';'+str(distance)+';'+str(learning_rate)+';'
                    +str(factor_neuronas)+';'+str(corrida)+';'+str(tiempo_entrenamiento)+';'+str(tiempo_resolucion)+';'+str(tiempo_total)+'\n')
    archivo.close()

def main():
    #num_max_iter = 100000 #número máximo de iteraciones
    #learning_rate = 0.8 # tasa de aprendizaje 
    #delta_learning_rate = 0.99997 #tasa de cambio de la tasa de aprendizaje
    #delta_n = 0.9997 #tasa de cambio de  n???? buscar después
    sensi_radio = 1 # sensibilidad del radio de BMU
    sensi_learning_rate = 0.001 #sensibilidad del learning rate
    #factor_neuronas= 8 #cantidad de neuronas por ciudad en la red neuronal
    plotear = False #Si es positivo creará un plot cada 1000 iteraciones y una de la ruta final
    runs = 10 #veces en que se corre el modelo por cada instancia
    #genration = 1 : normal, 2: circulo

    for instancia in ['qa194','uy734','ar9152','fi10639','it16862','ei8246','gr9882','ja9847','kz9976','mu1979','rw1621','vm22775','ym7663']:
        for num_max_iter in [5000,10000,20000,50000,100000]:
            for delta_learning_rate in [0.9997, 0.98, 0.95]:
                delta_n=delta_learning_rate
                for learning_rate in [0.9, 0.8, 0.7, 0.6]:
                    for factor_neuronas in [2,4,6,8]:
                        for generation in [1,2]:
                            for corrida in range(runs):
                                print('Corrida número', corrida+1)
                                inicio=time.time()
                                problem = read_tsp(instancia+'.tsp')
                                route = som(instancia,problem, num_max_iter,learning_rate, delta_learning_rate, delta_n,sensi_radio,sensi_learning_rate,factor_neuronas, generation, plotear)
                                tiempo_entrenamiento = time.time()-inicio
                                print('Tiempo de entrenamiento',tiempo_entrenamiento)
                                inicio2= time.time()
                                problem = problem.reindex(route)
                                distance = route_distance(problem)
                                tiempo_resolucion = time.time()-inicio2                            
                                print('Tiempo de enrutamiento ', tiempo_resolucion)
                                tiempo_total = tiempo_entrenamiento + tiempo_resolucion
                                write(instancia, num_max_iter, delta_learning_rate, delta_n, distance, learning_rate, factor_neuronas, corrida+1, 
                                        tiempo_entrenamiento, tiempo_resolucion,tiempo_total, generation)
                                print('Ruta encontrada de distancia {}'.format(distance))
                                print('Tiempo total de tiempo de resolución ', tiempo_total)

def som_from_outside(data, instancia="ar9152"):
    num_max_iter = 5000 #número máximo de iteraciones
    learning_rate = 0.8 # tasa de aprendizaje 
    delta_learning_rate = 0.99997 #tasa de cambio de la tasa de aprendizaje
    delta_n = 0.9997 #tasa de cambio de  n???? buscar después
    sensi_radio = 1 # sensibilidad del radio de BMU
    sensi_learning_rate = 0.001 #sensibilidad del learning rate
    factor_neuronas= 8 #cantidad de neuronas por ciudad en la red neuronal
    plotear = False #Si es positivo creará un plot cada 1000 iteraciones y una de la ruta final
    #genration = 1 : normal, 2: circulo
    generation = 2 
    route = som_2(instancia,data,num_max_iter,learning_rate,delta_learning_rate, delta_n,
                sensi_radio,sensi_learning_rate,factor_neuronas, generation, plotear)
    problem = problem.reindex(route)
    distance = route_distance(problem)
    print('Ruta encontrada de distancia {}'.format(distance))

import multiprocessing

def som_from_outside_2(data,return_list_per,return_list_dist,cluster,return_distancia):
    instancia = "ar9152"
    num_max_iter = 100 #número máximo de iteraciones
    learning_rate = 0.8 # tasa de aprendizaje 
    delta_learning_rate = 0.99997 #tasa de cambio de la tasa de aprendizaje
    delta_n = 0.9997 #tasa de cambio de  n???? buscar después
    sensi_radio = 1 # sensibilidad del radio de BMU
    sensi_learning_rate = 0.001 #sensibilidad del learning rate
    factor_neuronas= 8 #cantidad de neuronas por ciudad en la red neuronal
    plotear = False #Si es positivo creará un plot cada 1000 iteraciones y una de la ruta final
    #genration = 1 : normal, 2: circulo
    generation = 2 
    route = som(instancia,data,num_max_iter,learning_rate,delta_learning_rate, delta_n,
                sensi_radio,sensi_learning_rate,factor_neuronas, generation, plotear)
    data = data.reindex(route)
    distance = route_distance(data)
    #print('Ruta encontrada de distancia {}'.format(distance))
    return_list_dist[cluster] = route
    return_list_per[cluster] = data
    return_distancia[cluster] = distance

def som_2(instancia,problem, iterations, learning_rate,delta_learning_rate, delta_n,sensi_radio,sensi_learning_rate,factor_neuronas,generation,plotear):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtenemos primero las ciudades normalizadas (con coordenadas en [0,1])
    cities = problem.copy()
    #cities[['x', 'y']] = normalize(cities[['x', 'y']]) #para la cosa normal
    cities = normalize_2(cities)
    #La población de neuronas se crea con factor_neuronas veces la cantidad de ciudades
    n = cities.shape[0] * factor_neuronas

    # Generamos una adecuada red de neuronas de la forma
    if generation == 1:
        network = generate_network(n)
    elif generation ==2:
        network = generate_circular_network_center(n, cities)

    print('Red de {} neuronas creadas. Comenzando las iteraciones:'.format(n))

    if plotear:
        plot_network(cities, network, name='imagenes/'+instancia+'/{:05d}.png'.format(-1))

    for i in range(iterations):
        if not i % 100:
            print('\t> Iteración {}/{}'.format(i, iterations), end="\r")
        # Se escoge una ciudad de forma aleatoria
        city = cities.sample(1)[['x', 'y']].values
        #Se busca la neurona más cercana a la ciudad, la winner neuron
        winner_idx = select_closest(network, city)
        #Genera un filtro que aplica los cambios al winner o BMU
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        # Actualizar los pesos de la red según una distribución gaussiana
        network += gaussian[:,np.newaxis] * learning_rate * (city - network)
        
        # actualizar las parametros
        learning_rate = learning_rate * delta_learning_rate
        n = n * delta_n

        # Chequear para plotear cada 1000 iteraciones
        if plotear:
            if not i % 1000:
                plot_network(cities, network, name='imagenes/'+instancia+'/{:05d}.png'.format(i))

        # Chequear si algún parametro a caído por debajo de la sensibilidad
        if n < sensi_radio:
            print('Radio por debajo de sensibilidad, Se ha terminado la ejecución',
            'a {} las iteraciones'.format(i))
            break
        if learning_rate < sensi_learning_rate:
            print('Learning rate por debajo de sensibilidad, Se ha terminado la ejecución a las {} iteraciones'.format(i))
            break
    else:
        print('Se han completado las {} iteraciones.'.format(iterations))

    if plotear:
        plot_network(cities, network, name='imagenes/'+instancia+'/final.png')


    route = get_route(cities, network)
    
    if plotear:
        plot_route(cities, route, 'imagenes/'+instancia+'/route.png')
    return route

def som(instancia,problem, iterations, learning_rate,delta_learning_rate, delta_n,sensi_radio,sensi_learning_rate,factor_neuronas,generation,plotear):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtenemos primero las ciudades normalizadas (con coordenadas en [0,1])
    cities = problem.copy()
    cities[['x', 'y']] = normalize(cities[['x', 'y']]) 
    #La población de neuronas se crea con factor_neuronas veces la cantidad de ciudades
    n = cities.shape[0] * factor_neuronas

    # Generamos una adecuada red de neuronas de la forma
    if generation == 1:
        network = generate_network(n)
    elif generation ==2:
        network = generate_circular_network_center(n, cities)

    print('Red de {} neuronas creadas. Comenzando las iteraciones:'.format(n))

    if plotear:
        plot_network(cities, network, name='imagenes/'+instancia+'/{:05d}.png'.format(-1))

    for i in range(iterations):
        if not i % 100:
            print('\t> Iteración {}/{}'.format(i, iterations), end="\r")
        # Se escoge una ciudad de forma aleatoria
        city = cities.sample(1)[['x', 'y']].values
        #Se busca la neurona más cercana a la ciudad, la winner neuron
        winner_idx = select_closest(network, city)
        #Genera un filtro que aplica los cambios al winner o BMU
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        # Actualizar los pesos de la red según una distribución gaussiana
        network += gaussian[:,np.newaxis] * learning_rate * (city - network)
        
        # actualizar las parametros
        learning_rate = learning_rate * delta_learning_rate
        n = n * delta_n

        # Chequear para plotear cada 1000 iteraciones
        if plotear:
            if not i % 1000:
                plot_network(cities, network, name='imagenes/'+instancia+'/{:05d}.png'.format(i))

        # Chequear si algún parametro a caído por debajo de la sensibilidad
        if n < sensi_radio:
            print('Radio por debajo de sensibilidad, Se ha terminado la ejecución',
            'a {} las iteraciones'.format(i))
            break
        if learning_rate < sensi_learning_rate:
            print('Learning rate por debajo de sensibilidad, Se ha terminado la ejecución a las {} iteraciones'.format(i))
            break
    else:
        print('Se han completado las {} iteraciones.'.format(iterations))

    if plotear:
        plot_network(cities, network, name='imagenes/'+instancia+'/final.png')


    route = get_route(cities, network)
    
    if plotear:
        plot_route(cities, route, 'imagenes/'+instancia+'/route.png')
    return route

if __name__ == '__main__':
    main()
