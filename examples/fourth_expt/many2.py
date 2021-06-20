import math
import numpy as np

def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    #print 'Evaluating at (%f, %f)' % (x, y)

    # if x < 0 or x > 5.0 or y > 5.0:
    #     return np.nan
    # Feasible region: x in [0,5] and y in [0,5]

    obj1 = michalewicz(75, x, y)
    obj2 = michalewicz(100, x, y)
    obj3 = beale(x, y)
    obj4 = stybilinski(x, y)
	
    return {
        "michalewicz_75"        : obj1, 
        "michalewicz_100"       : obj2,
        "beale"                 : obj3,
        "stybilinski"           : obj4,
    }

def michalewicz(m,x,y):
    x = scale_from_unity(x,0,np.pi)
    y = scale_from_unity(y,0, np.pi)
    return -1*(np.sin(x)*np.sin(x**2/np.pi)**(2*m) + np.sin(y)*np.sin(2*y**2/np.pi)**(2*m))
   

def easom(x,y):
    x = scale_from_unity(x,-100, 100)
    y = scale_from_unity(y,-100, 100)
    return -np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2 - (y-np.pi)**2)

def beale(x,y):
    x = scale_from_unity(x,-4.5, 4.5)
    y = scale_from_unity(y,-4.5, 4.5)
    return ((1.5-x+x*y)*2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2)/10000

def stybilinski(x,y):
    x = scale_from_unity(x,-5, 5)
    y = scale_from_unity(y,-5, 5)
    return (x**4 + y**4 - 16*(x**2 + y**2) + 5*(x + y))/100


def gramacy(x,y):
    x = scale_from_unity(x, -2, 6)
    y = scale_from_unity(y, -2, 6)
    return x*np.exp(-x**2 -y**2)


def griewank(x,y):
    x = scale_from_unity(x, -600, 600)
    y = scale_from_unity(y, -600, 600)
    return (x**2 + y**2)/4000 - np.cos(x)*np.cos(y/np.sqrt(2))+1

def branin(x,y):
    x = scale_from_unity(x, -10, 10)
    y = scale_from_unity(y, -10, 10)

    return float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)

def levy(x,y):
    x = scale_from_unity(x, -5, 10)
    y = scale_from_unity(y, 0, 15)
    w1 = (x-1)/4 + 1
    w2 = (y-1)/4 + 1
    return np.square(np.sin(np.pi*w1)) + (w2-1)**2*(1 + np.sin(2* np.pi*w2)**2)

scale = lambda x, a, b, a2, b2: (x-a)*(b2-a2)/(b-a)+a2
def scale_from_unity(x, a, b):
    return scale(x, -1,1,a,b)


def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in parabolic.py'
        return np.nan
