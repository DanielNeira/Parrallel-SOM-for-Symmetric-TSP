import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
#Given three colinear points p, q, r, the function checks if 
#point q lies on line segment 'pr' 
def onSegment(p,q,r):
    #print("lo que entra en como p", p[0],"Lo que entra como r",r[0])
    #print("1r intentando en interseccion ",np.max(p[0],r[0]) )
    #print("1r intentando en interseccion ",min(p[0],r[0]))
    if (q[0]<= max(p[0],r[0]) and q[0] >= min(p[0],r[0]) 
        and q[1] <= max(p[1],r[1]) and q[1] >= min(p[1],r[1])):
        return True
    else:
        return False
  
# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 

def orientation(p,q,r):
    val = (q[1] - p[1])*(r[0] - q[0])-(q[0] - p[0])*(r[1] - q[1])
    if (val == 0):
        return 0 #colinear
    elif (val > 0):
        return 1 #clockwise
    else: # val < 0:
        return 2 #conterclock wise

# The main function that returns true if line segment 'p1q1' 
# and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2):
    # Find the four orientations needed for general and 
    # special cases 
    #print('p1:',p1,', q1:',q1,', p2:',p2,', q2:',q2)
    if False:
        plt.plot((p1[0],q1[0]),(p1[1],q1[1]), linewidth=3)
        plt.plot((p2[0],q2[0]),(p2[1],q2[1]), linewidth=3)
        plt.show()
        
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
    if (o1 != o2 and o3 != o4): # General case 
        return True
    # Special Casess
    elif (o1 == 0 and onSegment(p1, p2, q1)):
        return True   # p1, q1 and p2 are colinear and p2 lies on segment p1q1 
    # p1, q1 and q2 are colinear and q2 lies on segment p1q1 
    elif (o2 == 0 and onSegment(p1, q2, q1)): 
        return True 
    elif (o3 == 0 and onSegment(p2, p1, q2)):
        return True # p2, q2 and p1 are colinear and p1 lies on segment p2q2 
    elif (o4 == 0 and onSegment(p2, q1, q2)): 
        return True # p2, q2 and q1 are colinear and q1 lies on segment p2q2 
    else:
        return False # Doesn't fall in any of the above cases 
"""
for i in range(10):
    if i==0:
        p1 = (2,1)
        q1 = (5,3)
        p2 = (3,4)
        q2 = (5,1)
    else:
        p1 = np.random.rand(2)
        q1 = np.random.rand(2)
        p2 = np.random.rand(2)
        q2 = np.random.rand(2)

    if (doIntersect(p1, q1, p2, q2)):
        print("Si")
    else:
        print("No")
    plt.plot((p1[0],q1[0]),(p1[1],q1[1]), linewidth=3)
    plt.plot((p2[0],q2[0]),(p2[1],q2[1]), linewidth=3)
    plt.show()
"""