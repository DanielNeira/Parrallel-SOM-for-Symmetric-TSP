def write(instancia, num_max_iter, delta_learning_rate, delta_n, distance, learning_rate, factor_neuronas, corrida)
    """
    Esta función toma los strings motor, solucion, x_sol, tiemp_ejecutado, y si plot es True grafica la sol
    primera columna: instancia, segunda: función objetivo y tercera: tiemp_ejecucion
    """ 
    archivo = open('resultados.txt','a')
    archivo.write(str(instancia)+';'+str(num_max_iter)+';'+str(delta_learning_rate)+';'+str(delta_n)+';'+str(distance)+';'+str(learning_rate)+';'+str(factor_neuronas)+';'+str(corrida)+'\n')
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

    for instancia in ['qa194','uy734','ar9152','fi10639','it16862']:#'ch71009']:
        for num_max_iter in [5000,10000,20000,50000,100000]:
            for delta_learning_rate,delta_n in [0.9997, 0.98, 0.95]:
                delta_n=delta_learning_rate
                for learning_rate in [0.9, 0.8, 0.7, 0.6]:
                    for factor_neuronas in [2,4,6,8]:
                        for corrida in range(runs):
                            print('Corrida ', corrida+1)
                            problem = read_tsp(instancia+'.tsp')
                            route = som(instancia,problem, num_max_iter,learning_rate, delta_learning_rate, delta_n,sensi_radio,sensi_learning_rate,factor_neuronas, plotear)
                            problem = problem.reindex(route)
                            distance = route_distance(problem)
                            write(instancia, num_max_iter, delta_learning_rate, delta_n, distance, learning_rate, factor_neuronas, corrida+1)
                            print('Ruta encontrada de distancia {}'.format(distance))