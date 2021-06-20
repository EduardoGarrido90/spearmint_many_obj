import numpy as np
from matplotlib import pyplot as plt
# we use the following for plotting figures in jupyter
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Various distance definitions 

#A
def npoints_more_than(m1, m2, delta = 0):
    return (np.abs(m1-m2)>delta).sum()
#B
def percentage_more_than(m1, m2, delta = 0):
    return 100*(np.abs(m1-m2)>delta).sum()/len(m1)
#C
def sum_difference(m1, m2, delta = 0):
    return np.abs(m1-m2)[np.where(np.abs(m1-m2) > delta)].sum()
#F delta not needed. Just added to fit the mean_dist method called from below
def max_difference(m1, m2, delta = 0):
    return np.amax(np.abs(m1-m2))
#G delta not needed
def avg_difference(m1,m2, delta = 0):
    return np.abs(m1-m2).sum()/len(m1)
# Error cuadratico
def quad_error(m1,m2, delta=0):
    return np.linalg.norm(m1-m2)
#I error medio relativo (a la imagen)
def avg_rel_difference(m1,m2, delta = 0):
    return (np.abs(m1-m2).sum()/len(m1))/(np.amax([m1,m2])-np.amin([m1,m2]))


# Transformations
from scipy.optimize import curve_fit

#line = lambda x, a, b: a*x + b
def line(x, a, b):
    return a * x + b

#finds best a, b such that gp2 = a*gp1 + b
def line_transform(m1, m2, X):
    popt, pcov = curve_fit(line, m1, m2, bounds = ([0,-np.inf], np.inf)) # a must be positive
    #print(popt)
    Tm1 = line(m1, popt[0], popt[1])
    return Tm1


'''
* gp1, gp2 are the processes to be predicted over the domain X
* epsilon1 \in [0,1] is the weight that will be given in the final distance to the distance between gp1 and gp2 means
  epsilon2 is the weight thatll be given to the correlation between the means
  1 - epsilon1 - epsilon2 will be the weight given to the distance between their variance
* transformation is a function with respect to which we decide if mean1 and mean2 are similar.
i.e., distance(mean1, mean2) = 0 iff mean2 = transformation(m1). (transformation could be calculated as a curve fit)
* var_dist and mean_dist are functions to define the distance between both mean vectors/ variance vectors
* delta is a number used as an argument in mean_dist to decide when the distance between two values is negligible. 
e.g., mean_dist(m1,m2) = sum(|m1-m2| when |m1-m2|> delta)
'''
def distance(gp1, gp2, X, 
             epsilon1 = 0.25, epsilon2 = 0.75, delta = 0, 
             transformation = line_transform, 
             mean_dist = avg_rel_difference, 
             var_dist = np.linalg.norm,
             ret_intermediate = False):
    intermediate = ""
    m1, v1 = gp1.predict(X, full_cov = False)
    m2, v2 = gp2.predict(X, full_cov = False)
    #These flattenings were necessary when using GPy gps, but are not when using models.gp
    #m1= m1.flatten()
    #m2 = m2.flatten()
    Tm1 = transformation(m1, m2, X)
    d1 = mean_dist(Tm1, m2, delta)
    intermediate = intermediate + "\tdistance between means, " + str(round(d1,5))
    corrmat = np.corrcoef(m1, m2)    
    ro = (corrmat[1][0]).round(5)    
    intermediate = intermediate + "\n\tcorrelat between means, " + str(ro)
    d2 =  1-max(0,ro) # Negative correlation is "as bad as" zero correlation for us.
    #This might be change if we manage maximizing gp1 and minimizing gp2, in which case ro = -1 would be good
    d3 = var_dist(v1-v2)    
    intermediate = intermediate + "\n\tdistance between vars, " + str(round(d3,5))
    total_distance = epsilon1*d1 + epsilon2*d2 + (1-epsilon1 - epsilon2)*d3
    if ret_intermediate == True:
        return (total_distance, intermediate)
    else:
        return total_distance 
    
