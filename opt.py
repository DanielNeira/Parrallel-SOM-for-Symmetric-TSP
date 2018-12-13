import numpy as np
import pandas as pd

def euclidean_distance(a, b):
    """Return the array of distances of two numpy arrays of points."""
    #print(a,b,np.linalg.norm(a - b))
    return np.linalg.norm(a - b)

def reverse_segment_if_better(tour, i, j, k, distancia_inicial):
    base = distancia_inicial - (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[i+1])
                                + euclidean_distance(tour[['x','y']].iloc[j],tour[['x','y']].iloc[j+1])
                                + euclidean_distance(tour[['x','y']].iloc[k],tour[['x','y']].iloc[k+1]))
    
    d1 = base + (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[k])  #AcbD
                + euclidean_distance(tour[['x','y']].iloc[j+1],tour[['x','y']].iloc[j])
                + euclidean_distance(tour[['x','y']].iloc[i+1],tour[['x','y']].iloc[k+1]))
    
    d2 = base + (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[i+1])  #ABcD
                + euclidean_distance(tour[['x','y']].iloc[j],tour[['x','y']].iloc[k])
                + euclidean_distance(tour[['x','y']].iloc[j+1],tour[['x','y']].iloc[k+1]))
    
    d3 = base + (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[j])  #AbCD
                + euclidean_distance(tour[['x','y']].iloc[i+1],tour[['x','y']].iloc[j+1])
                + euclidean_distance(tour[['x','y']].iloc[k],tour[['x','y']].iloc[k+1]))
    
    d4 = base + (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[j])  #AbcD
                + euclidean_distance(tour[['x','y']].iloc[i+1],tour[['x','y']].iloc[k])
                + euclidean_distance(tour[['x','y']].iloc[j+1],tour[['x','y']].iloc[k+1]))
    
    d5 = base + (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[k])  #AcBD
                + euclidean_distance(tour[['x','y']].iloc[j+1],tour[['x','y']].iloc[i+1])
                + euclidean_distance(tour[['x','y']].iloc[j],tour[['x','y']].iloc[k+1]))
    
    d6 = base + (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[j+1])  #ACbD
                + euclidean_distance(tour[['x','y']].iloc[k],tour[['x','y']].iloc[j])
                + euclidean_distance(tour[['x','y']].iloc[i+1],tour[['x','y']].iloc[k+1]))
    
    d7 = base + (euclidean_distance(tour[['x','y']].iloc[i],tour[['x','y']].iloc[j+1])  #ACBD
                + euclidean_distance(tour[['x','y']].iloc[k],tour[['x','y']].iloc[i+1])
                + euclidean_distance(tour[['x','y']].iloc[j],tour[['x','y']].iloc[k+1]))

    minimo = np.argmin([distancia_inicial,d1,d2,d3,d4,d5,d6,d7])

    if minimo == 1:
        return tour.iloc[:i+1].append(tour.iloc[j+1:k+1].iloc[::-1], 
                                    ignore_index = True).append(tour.iloc[i+1:j+1].iloc[::-1], 
                                                                ignore_index = True).append(tour.iloc[k+1:],
                                                                                            ignore_index = True), d1
    elif minimo == 2:
        return tour.iloc[:i+1].append(tour.iloc[i+1:j+1], 
                                    ignore_index = True).append(tour.iloc[j+1:k+1].iloc[::-1], 
                                                                ignore_index = True).append(tour.iloc[k+1:],
                                                                                            ignore_index = True), d2
    elif minimo == 3:
        return tour.iloc[:i+1].append(tour.iloc[i+1:j+1].iloc[::-1],
                                    ignore_index = True).append(tour.iloc[j+1:k+1],
                                                                ignore_index = True).append(tour.iloc[k+1:], 
                                                                                            ignore_index = True), d3
    elif minimo == 4:
        return tour.iloc[:i+1].append(tour.iloc[i+1:j+1].iloc[::-1],
                                    ignore_index = True).append(tour.iloc[j+1:k+1].iloc[::-1],
                                                                ignore_index = True).append(tour.iloc[k+1:],
                                                                                            ignore_index = True), d4
    elif minimo == 5:
        return tour.iloc[:i+1].append(tour.iloc[j+1:k+1].iloc[::-1],
                                    ignore_index = True).append(tour.iloc[i+1:j+1],
                                                                ignore_index = True).append(tour.iloc[k+1:],
                                                                                            ignore_index = True), d5
    elif minimo == 6:
        return tour.iloc[:i+1].append(tour.iloc[j+1:k+1],
                                    ignore_index = True).append(tour.iloc[i+1:j+1].iloc[::-1],
                                                                ignore_index = True).append(tour.iloc[k+1:],
                                                                                            ignore_index = True), d6
    elif minimo == 7:
        return tour.iloc[:i+1].append(tour.iloc[j+1:k+1],
                                    ignore_index = True).append(tour.iloc[i+1:j+1],
                                                                ignore_index = True).append(tour.iloc[k+1:],
                                                                                            ignore_index = True), d7
    else:
        return tour,distancia_inicial
    return print("ERROOOOOOOOOOR")

def three_opt(tour_in,distancia_inicial,vecindario):
    "Iterative improvement based on 3 exchange."
    tour_in = tour_in.reset_index(drop=True)
    tour_in = tour_in.append(tour_in.iloc[0], ignore_index = True)
    tour_in = tour_in.reset_index(drop=True)
    N = len(tour_in)-2
    for a in range(N):
        ruptura = False
        for b in range(a+2,N):
            if b <= a+2+vecindario:
                for c in range(b+2,N+(a>0)):
                    if c <= b+2+vecindario:
                        print(a,b,c, distancia_inicial)
                        tour_out, distancia = reverse_segment_if_better(tour_in, a, b, c, distancia_inicial)
                        if (distancia < distancia_inicial):
                            distancia_inicial = distancia
                            tour_in = tour_out.copy()
                            ruptura = True
                            break
                    if ruptura:
                        break
    return tour_in, distancia_inicial
