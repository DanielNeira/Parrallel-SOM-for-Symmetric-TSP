from multiprocessing import Process
import main as mn
import numpy as np
import time

class ProcessParallel(object):
    """
    To Process the  functions parallely

    """    
    def __init__(self, *jobs):
        """
        """
        self.jobs = jobs
        self.processes = []

    def fork_processes(self):
        """
        Creates the process objects for given function deligates
        """
        for job in self.jobs:
            proc  = Process(target=job)
            self.processes.append(proc)

    def start_all(self):
        """
        Starts the functions process all together.
        """
        for proc in self.processes:
            proc.start()

    def join_all(self):
        """
        Waits untill all the functions executed.
        """
        for proc in self.processes:
            proc.join()

def multiply(a, b):
    print(a*b)
    #return a * b


"""
#How to run:
procs = ProcessParallel(multiply) #Add all the process in list
procs.fork_processes() #starts process execution 
procs.start_all() #wait until all the process got executed
procs.join_all()
"""
threads = list()
datos = np.random.rand(10,2)
for i in range(10):
    t = Process(target=multiply,args=datos[i])
    threads.append(t)
    t.start()
    t.join()
