import math
import numpy as np

def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    #print 'Evaluating at (%f, %f)' % (x, y)

    # if x < 0 or x > 5.0 or y > 5.0:
    #     return np.nan
    # Feasible region: x in [0,5] and y in [0,5]

    obj1 = 30*(x**4 + y**4)
    obj2 = 30*(x**2 + y**2)
    obj3 = griewank(x, y)
    obj4 = gramacy(x, y)
	
    return {
        "parabola6"       : obj1, 
        "parabola2"       : obj2,
        "griewank"       : obj3,
        "gramacy"        : obj4,
    }



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
