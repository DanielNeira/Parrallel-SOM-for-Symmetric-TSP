import pandas as pd
import numpy as np

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
        return cities

def normalize(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)

def normalize_2(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    #print(np.shape(points))
    #print(points,"\n\n",points[0],"\n\n",np.max(np.asarray(points)[:,0]))
    points = np.asarray(points)
    ratio = (max(points[:,0]) - min(points[:,0])) / (max(points[:,1]) - min(points[:,1])), 1
    ratio = np.array(ratio) / max(ratio)
    #norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    norm = [(i - i.min()) / (i.max() - i.min()) for i in range(points)]
    #return norm.apply(lambda p: ratio * p, axis=1)
    return [ratio[i]*i for i in range(norm)]
