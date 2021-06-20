import math
import numpy as np
import time

def evaluate(job_id, params):

    x = params['X']
    y = params['Y']

    #print 'Evaluating at (%f, %f)' % (x, y)

    # if x < 0 or x > 5.0 or y > 5.0:
    #     return np.nan
    # Feasible region: x in [0,5] and y in [0,5]
    
    t1 = time()
    obj3 = branin(x,y) 
    t2 = time()
    obj2 = -branin(x,y)
    t3 = time()
    obj1 = 3*branin(x,y)
	t4 = time()
    
    #with open('~/Spearmint/expt_05_10/times.txt', 'w') as fd:
    #    f.write(string(t2-t1) + "\t\t\t" + string(t3-t2) + "\t\t\t" + string(t4-t3) + "\n")
    
    return {
        "branin_1"       : obj1, 
        "branin_2"       : obj2,
        "branin_3"       : obj3,
    }

def branin(x,y):
    return float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)


def main(job_id, params):
    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print ex
        print 'An error occurred in branin_con.py'
        return np.nan
