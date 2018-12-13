import numpy as np

from distance import select_closest

def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)

def generate_circular_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional circular points in the interval [0,1].
    """   
    network = np.zeros((size,2))
    for i in range(size):
        for j in range(2):
            if j == 0:
                network[i,j] = np.cos(i*2*np.pi/float(size))
            else:
                network[i,j] = np.sin(i*2*np.pi/float(size))
    return network

def generate_circular_network_center(size,cities):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional circular center in the center of mass of the cities.
    """   
    x_center = np.sum(cities['x'])/size
    y_center = np.sum(cities['y'])/size 

    network = np.zeros((size,2))
    
    for i in range(size):
        for j in range(2):
            if j == 0:
                network[i,j] = np.cos(i*2*np.pi/float(size))+x_center
            else:
                network[i,j] = np.sin(i*2*np.pi/float(size))+y_center
    return network   

def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))

def get_route(cities, network):
    """Return the route computed by a network."""
    cities['winner'] = cities[['x', 'y']].apply(
                            lambda c: select_closest(network, c),
                            axis=1, raw=True)

    return cities.sort_values('winner').index
