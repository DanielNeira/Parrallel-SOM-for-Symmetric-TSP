
import numpy as np

from io_helper import read_tsp, normalize
from neuron import generate_network, get_neighborhood, get_route, generate_circular_network, generate_circular_network_center
from distance import select_closest, euclidean_distance, route_distance
from plot import plot_network, plot_route

def main():
    num_max_iter = 100000 #número máximo de iteraciones
    learning_rate = 0.8 # tasa de aprendizaje 
    delta_learning_rate = 0.99997 #tasa de cambio de la tasa de aprendizaje
    delta_n = 0.9997 #tasa de cambio de  n???? buscar después
    sensi_radio = 1 # sensibilidad del radio de BMU
    sensi_learning_rate = 0.001 #sensibilidad del learning rate
    factor_neuronas= 8 #cantidad de neuronas por ciudad en la red neuronal
    plotear = True #Si es positivo creará un plot cada 1000 iteraciones y una de la ruta final
    runs = 10 #veces en que se corre el modelo por cada instancia

    for instancia in ['qa194']:#,'uy734','ar9152','fi10639','it16862','ch71009']:
        problem = read_tsp(instancia+'.tsp')
        #print(problem, type(problem))
        route = som(instancia,problem, num_max_iter,learning_rate, delta_learning_rate, delta_n,sensi_radio,sensi_learning_rate,factor_neuronas, plotear)
        problem = problem.reindex(route)
        distance = route_distance(problem)
        print('Ruta encontrada de distancia {}'.format(distance))


def som(instancia,problem, iterations, learning_rate,delta_learning_rate, delta_n,sensi_radio,sensi_learning_rate,factor_neuronas,plotear):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtenemos primero las ciudades normalizadas (con coordenadas en [0,1])
    cities = problem.copy()
    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    #La población de neuronas se crea con factor_neuronas veces la cantidad de ciudades
    n = cities.shape[0] * factor_neuronas

    # Generamos una adecuada red de neuronas de la forma
    #network = generate_network(n)
    #network = generate_circular_network(n)
    network = generate_circular_network_center(n,cities)

    if plotear:
        print('Red de {} neuronas creadas. Comenzando las iteraciones:'.format(n))

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
            print('Learning rate por debajo de sensibilidad, Se ha terminado la ejecución',
            'a las {} iteraciones'.format(i))
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
