#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
import GPy
import distance as d
from matplotlib import pyplot as plt


# ## Primero: 3 bowl shaped

# In[221]:





# In[228]:


nsamples = 20
x = np.linspace(-1, 1, nsamples)
y = np.linspace(-1, 1, nsamples)
X, Y = np.meshgrid(x, y)
XY = np.stack((X.flatten(), Y.flatten()), axis = -1)

npredict = 25
xnew = np.linspace(-1, 1, npredict)
ynew = np.linspace(-1, 1, npredict)
Xnew, Ynew = np.meshgrid(xnew, ynew)
XYnew = np.stack((Xnew.flatten(), Ynew.flatten()), axis = -1)

# GP predictions
Z1 = d.styb_tang(X, Y)
Z1_= Z1.reshape(Z1.size, 1) # shape necesaria para GPy
k1 = GPy.kern.RBF(2, variance=2., lengthscale=0.6, name="rbf")
gp1 = GPy.models.GPRegression(XY, Z1_, k1)
m1, v1 = gp1.predict_noiseless(XYnew, full_cov = True)

Z2 = d.ellipsoid(X, Y)
Z2_= Z2.reshape(Z2.size, 1) # shape necesaria para GPy
k2 = GPy.kern.RBF(2, variance=15., lengthscale=1, name="rbf")
gp2 = GPy.models.GPRegression(XY, Z2_, k2)
m2, v2 = gp2.predict_noiseless(XYnew, full_cov = True)

Z3 = d.sphere(X, Y)
Z3_= Z3.reshape(Z3.size, 1) # shape necesaria para GPy
k3 = GPy.kern.RBF(2, variance=7., lengthscale=1, name="rbf")
gp3 = GPy.models.GPRegression(XY, Z3_, k3)
m3, v3 = gp3.predict_noiseless(XYnew, full_cov = True)

# Real functions plot
fig = plt.figure(figsize=(15,5))

'''print("Real functions vs its prediction- bowl shaped edition")


plt.subplot(2, 3, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z1, cmap='plasma')
ax.set_zlim(np.min(Z1)-2,np.max(Z1)+2)
plt.title(d.styb_tang.__name__)

plt.subplot(2, 3, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z2, cmap='plasma')
ax.set_zlim(np.min(Z2)-2,np.max(Z2)+2)
plt.title(d.ellipsoid.__name__)

plt.subplot(2, 3, 3, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z3, cmap='plasma')
ax.set_zlim(np.min(Z3)-2,np.max(Z3)+2)
plt.title(d.sphere.__name__)'''

# GP plots

#plt.subplot(2, 3, 4, projection='3d')
plt.subplot(1, 3, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m1.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m1)-2,np.max(m1)+2)
plt.title(d.styb_tang.__name__)

#plt.subplot(2, 3, 5, projection='3d')
plt.subplot(1, 3, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m2.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m2)-2,np.max(m2)+2)
plt.title(d.ellipsoid.__name__)

#plt.subplot(2, 3, 6, projection='3d')
plt.subplot(1, 3, 3, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m3.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m3)-2,np.max(m3)+2)
plt.title(d.sphere.__name__)

plt.savefig('bowl.svg', format='svg', dpi=1200)


# In[229]:


distance_info = " "

    distance, intermediate = d.distance(gp1, gp2, XYnew, epsilon1 = 0.25, epsilon2 = 0.75, delta = 0, mean_dist =d.avg_rel_difference, ret_intermediate = True)
distance_info = distance_info + "\nstyb vs ellipsoid: \n" + intermediate
distance_info = distance_info + "\n\tsimilarity measure    , " + str(round(distance, 5)) + "\n\n"

distance, intermediate = d.distance(gp2, gp3, XYnew, epsilon1 = 0.25, epsilon2 = 0.75, delta = 0, mean_dist =d.avg_rel_difference, ret_intermediate = True)
distance_info = distance_info + "\nellipsoid vs sphere: \n" + intermediate
distance_info = distance_info + "\n\tsimilarity measure    , " + str(round(distance, 5)) + "\n\n"

print(distance_info)


# ## Ackleys

# In[ ]:





# In[26]:


nsamples = 20
x = np.linspace(-1, 1, nsamples)
y = np.linspace(-1, 1, nsamples)
X, Y = np.meshgrid(x, y)
XY = np.stack((X.flatten(), Y.flatten()), axis = -1)

npredict = 25
xnew = np.linspace(-1, 1, npredict)
ynew = np.linspace(-1, 1, npredict)
Xnew, Ynew = np.meshgrid(xnew, ynew)
XYnew = np.stack((Xnew.flatten(), Ynew.flatten()), axis = -1)

# GP predictions
a,b,c  = 20,0.2,2*np.pi

Z1 = d.ackley(20, 0.2,np.pi,X, Y)
Z1_= Z1.reshape(Z1.size, 1) # shape necesaria para GPy
k1 = GPy.kern.RBF(2, variance=10., lengthscale=0.1, name="rbf")
gp1 = GPy.models.GPRegression(XY, Z1_, k1)
m1, v1 = gp1.predict_noiseless(XYnew, full_cov = True)

Z3 = d.ackley(20, 0.2, 4*np.pi, X, Y)
Z3_= Z3.reshape(Z3.size, 1) # shape necesaria para GPy
k3 = GPy.kern.RBF(2, variance=4, lengthscale=0.1, name="rbf")
gp3 = GPy.models.GPRegression(XY, Z3_, k3)
m3, v3 = gp3.predict_noiseless(XYnew, full_cov = True)

Z4 = d.ackley(70, b, c, X, Y)
Z4_= Z4.reshape(Z4.size, 1) # shape necesaria para GPy
k4 = GPy.kern.RBF(2, variance=10, lengthscale=0.06, name="rbf")
gp4 = GPy.models.GPRegression(XY, Z4_, k4)
m4, v4 = gp4.predict_noiseless(XYnew, full_cov = True)

Z5 = d.ackley(100, b, c, X, Y)
Z5_= Z5.reshape(Z5.size, 1) # shape necesaria para GPy
k5 = GPy.kern.RBF(2, variance=10, lengthscale=0.07, name="rbf")
gp5 = GPy.models.GPRegression(XY, Z5_, k5)
m5, v5 = gp5.predict_noiseless(XYnew, full_cov = True)

# Real functions plot
fig = plt.figure(figsize=(12,5))
'''
print("Real functions vs its prediction- bowl shaped edition")

plt.subplot(2, 4, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z1, cmap='plasma')
ax.set_zlim(np.min(Z1)-2,np.max(Z1)+2)

plt.subplot(2, 4, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z3, cmap='plasma')
ax.set_zlim(np.min(Z3)-2,np.max(Z3)+2)

plt.subplot(2, 4, 3, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z4, cmap='plasma')
ax.set_zlim(np.min(Z4)-2,np.max(Z4)+2)

plt.subplot(2, 4, 4, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z5, cmap='plasma')
ax.set_zlim(np.min(Z5)-2,np.max(Z5)+2)
'''
# GP plots

plt.subplot(1, 4, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m1.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m1)-2,np.max(m1)+2)
plt.title('a = 20, b = 0.2, c = pi')

plt.subplot(1, 4, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m3.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m3)-2,np.max(m3)+2)
plt.title('a = 20, b = 0.2, c = 4pi')

plt.subplot(1, 4, 3, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m4.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m4)-2,np.max(m4)+2)
plt.title('a = 70, b = 0.2, c = 2pi')

plt.subplot(1, 4, 4, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m5.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m5)-2,np.max(m5)+2)
plt.title('a = 100, b = 0.2, c = 2pi')

#plt.savefig('ackley.svg', format='svg', dpi=1200)


# In[238]:


distance_info = " "

distance, intermediate = d.distance(gp1, gp3, XYnew, epsilon1 = 0.25, epsilon2 = 0.75, delta = 0, mean_dist =d.avg_rel_difference, ret_intermediate = True)
distance_info = distance_info + "\nc = pi vs c = 4pi: \n" + intermediate
distance_info = distance_info + "\n\tsimilarity measure    , " + str(round(distance, 5)) + "\n\n"

distance, intermediate = d.distance(gp4, gp5, XYnew, epsilon1 = 0.25, epsilon2 = 0.75, delta = 0, mean_dist =d.avg_rel_difference, ret_intermediate = True)
distance_info = distance_info + "\na = 70 vs a = 100: \n" + intermediate
distance_info = distance_info + "\n\tsimilarity measure    , " + str(round(distance, 5)) + "\n\n"

print(distance_info)


# 
# ## Nothing to do

# In[30]:


# GP predictions
Z1 = d.griewank(X, Y)
Z1_= Z1.reshape(Z1.size, 1) # shape necesaria para GPy
k1 = GPy.kern.RBF(2, variance=4., lengthscale=0.1, name="rbf")
gp1 = GPy.models.GPRegression(XY, Z1_, k1)
m1, v1 = gp1.predict_noiseless(XYnew, full_cov = True)

Z2 = d.levy(X, Y)
Z2_= Z2.reshape(Z2.size, 1) # shape necesaria para GPy
k2 = GPy.kern.RBF(2, variance=15., lengthscale=0.1, name="rbf")
gp2 = GPy.models.GPRegression(XY, Z2_, k2)
m2, v2 = gp2.predict_noiseless(XYnew, full_cov = True)

print(np.linalg.norm(np.diag(v1)-np.diag(v2)))
# Real functions plot
fig = plt.figure(figsize=(15,5))

'''
print("Real functions vs its prediction- bowl shaped edition")


plt.subplot(2, 2, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z1, cmap='plasma')
ax.set_zlim(np.min(Z1)-2,np.max(Z1)+2)
plt.title(d.griewank.__name__)

plt.subplot(2, 2, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z2, cmap='plasma')
ax.set_zlim(np.min(Z2)-2,np.max(Z2)+2)
plt.title(d.levy.__name__)
'''
# GP plots

#plt.subplot(2, 2, 3, projection='3d')
plt.subplot(1, 2, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m1.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m1)-2,np.max(m1)+2)
plt.title(d.griewank.__name__)

#plt.subplot(2, 2, 4, projection='3d')
plt.subplot(1, 2, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m2.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m2)-2,np.max(m2)+2)
plt.title(d.levy.__name__)

#plt.savefig('misc.svg', format='svg', dpi=1200)


# In[232]:


distance_info = " "

distance, intermediate = d.distance(gp1, gp2, XYnew, epsilon1 = 0.25, epsilon2 = 0.75, delta = 0, mean_dist =d.avg_rel_difference, ret_intermediate = True)
distance_info = distance_info + "\ngriewank vs levy: \n" + intermediate
distance_info = distance_info + "\n\tsimilarity measure    , " + str(round(distance, 5)) + "\n\n"

print(distance_info)


# ## 1D

# In[24]:


# defino michalewicz para d=1
def griewank(x):
    #domain is [-600,600]
    x = d.scale_from_unity(x,-600,600)
    return x**2/4000-np.cos(x)+1
griewank.__name__ = "grwnk"

nsamples = 500
X = np.linspace(-1, 1, nsamples)[:, None]
npredict = 1000
Xnew = np.linspace(-1, 1, npredict)[:, None]

Z1 = d.michalewicz(50, X)
k1 = GPy.kern.RBF(1, variance=1, lengthscale=0.05, name="rbf")
gp1 = GPy.models.GPRegression(X, Z1, k1)
m1, v1 = gp1.predict_noiseless(Xnew, full_cov = True)

Z2 = d.michalewicz(100, X)
k2 = GPy.kern.RBF(1, variance=1, lengthscale=0.03, name="rbf")
gp2 = GPy.models.GPRegression(X, Z2, k2)
m2, v2 = gp2.predict_noiseless(Xnew, full_cov = True)

Z3 = X**2
k3 = GPy.kern.RBF(1, variance=20., lengthscale=0.1, name="rbf")
gp3 = GPy.models.GPRegression(X, Z3, k3)
m3, v3 = gp3.predict_noiseless(Xnew, full_cov = True)

fig = plt.figure(figsize=(20,5))
'''
plt.subplot(2, 2, 1)
plt.plot(X, Z1)
plt.plot(X, Z2)

plt.subplot(2, 2, 2)
plt.plot(X, Z3)
'''
plt.subplot(1, 2, 1)
plt.plot(Xnew, m1)
plt.plot(Xnew, m2)
plt.legend(labels = ["m = 50", "m = 100"])
plt.title('Michalewicz')

plt.subplot(1,2,2)
plt.plot(Xnew, m3)
plt.title('X^2')
#plt.savefig('1d.svg', format='svg', dpi=1200)


# In[226]:


distance_info = " "

distance, intermediate = d.distance(gp1, gp2, XYnew, epsilon1 = 0.25, epsilon2 = 0.75, delta = 0.0, mean_dist =d.avg_rel_difference, ret_intermediate = True)
distance_info = distance_info + "\nMichalewicz m=50 vs m=100: \n" + intermediate
distance_info = distance_info + "\n\tsimilarity measure    , " + str(round(distance, 5)) + "\n\n"

distance, intermediate = d.distance(gp2, gp3, XYnew, epsilon1 = 0.25, epsilon2 = 0.75, delta = 0, mean_dist =d.avg_rel_difference, ret_intermediate = True)
distance_info = distance_info + "\nMichalewicz vs X^2: \n" + intermediate
distance_info = distance_info + "\n\tsimilarity measure    , " + str(round(distance, 5)) + "\n\n"

print(distance_info)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Estos me los pidio Eduardo en un inicio
# 
# ## Booth (2D)

# In[ ]:





# In[4]:


# Para hacer regresión

# Samples
nsamples = 10
x = np.linspace(-1, 1, nsamples)

X, Y = np.meshgrid(x, x)
Z = d2.booth(X, Y)

# Puntos a predecir
npredict = 50
xnew = np.linspace(-1, 1, npredict)
ynew = np.linspace(-1, 1, npredict)
Xnew, Ynew = np.meshgrid(xnew, ynew)

# Cambiamos formato de datos para hacer regresion
XY = np.stack((X.flatten(), Y.flatten()), axis = -1)
Z = Z.reshape(Z.size, 1)
XYnew = np.stack((Xnew.flatten(), Ynew.flatten()), axis = -1)

# Covariance between training sample points
k1 = GPy.kern.RBF(2, variance=1., lengthscale=0.8, name="rbf")
k2 = GPy.kern.RBF(2, variance=1., lengthscale=0.5, name="rbf")
k3 = GPy.kern.RBF(2, variance=1., lengthscale=0.1, name="rbf")

'''K1xx = k1.K(X,X) 
K2xx = k2.K(X,X) 
'''
gp1 = GPy.models.GPRegression(XY, Z, k1)
#mean1, Cov1 = gp1.predict_noiseless(XYnew, full_cov=True)

gp2 = GPy.models.GPRegression(XY, Z, k2)
#mean2, Cov2 = gp2.predict_noiseless(XYnew, full_cov=True)

gp3 = GPy.models.GPRegression(XY, Z, k3)
#mean3, Cov3 = gp3.predict_noiseless(XYnew, full_cov=True)

gp1.plot(title = 'lengthscale = 0.8')
gp2.plot(title = 'lengthscale = 0.5')
gp3.plot(title = 'lengthscale = 0.1')


# In[5]:


print('\n1')
print("\nDistance between l = 0.8 and l = 0.5:\t", d2.distance(gp1, gp2, XYnew, mean_dist = d2.avg_rel_difference).round(4))
print('\n2')
print("\nDistance between l = 0.5 and l = 0.1:\t", d2.distance(gp2, gp3, XYnew, mean_dist = d2.avg_rel_difference).round(4))
print('\n3')
print("\nDistance between l = 0.8 and l = 0.1:\t", d2.distance(gp1, gp3, XYnew, mean_dist = d2.avg_rel_difference).round(4))


# ## Seno 

# In[19]:


# Para hacer regresión

# Samples
nsamples = 20
X = np.linspace(-1, 1, nsamples)[:, None]
Z = np.sin(X)

# Puntos a predecir
npredict = 100
Xnew = np.linspace(-1, 1, npredict)[:, None]

# Covariance between training sample points
k1 = GPy.kern.RBF(1, variance=1., lengthscale=0.8, name="rbf")
k2 = GPy.kern.RBF(1, variance=1., lengthscale=0.5, name="rbf")
k3 = GPy.kern.RBF(1, variance=1., lengthscale=0.1, name="rbf")

K1xx = k1.K(X,X) 
K2xx = k2.K(X,X) 

gp1 = GPy.models.GPRegression(X, Z, k1)
#mean1, Cov1 = gp1.predict_noiseless(XYnew, full_cov=True)

gp2 = GPy.models.GPRegression(X, Z, k2)
#mean2, Cov2 = gp2.predict_noiseless(XYnew, full_cov=True)

gp3 = GPy.models.GPRegression(X, Z, k3)
#mean3, Cov3 = gp3.predict_noiseless(XYnew, full_cov=True)

gp1.plot(title = 'lengthscale = 0.8')
gp2.plot(title = 'lengthscale = 0.5')
gp3.plot(title = 'lengthscale = 0.1')

print('\n1')
print("\nDistance between l = 0.8 and l = 0.5:\t", d2.distance(gp1, gp2, XYnew, mean_dist = d2.avg_rel_difference).round(4))
print('\n2')
print("\nDistance between l = 0.5 and l = 0.1:\t", d2.distance(gp2, gp3, XYnew, mean_dist = d2.avg_rel_difference).round(4))
print('\n3')
print("\nDistance between l = 0.8 and l = 0.1:\t", d2.distance(gp1, gp3, XYnew, mean_dist = d2.avg_rel_difference).round(4))


# ## Seno con más frecuencia

# In[20]:


# Para hacer regresión

# Samples
nsamples = 20
X = np.linspace(-1, 1, nsamples)[:, None]
Z = np.sin(5*X)

# Puntos a predecir
npredict = 100
Xnew = np.linspace(-1, 1, npredict)[:, None]

# Covariance between training sample points
k1 = GPy.kern.RBF(1, variance=1., lengthscale=0.8, name="rbf")
k2 = GPy.kern.RBF(1, variance=1., lengthscale=0.5, name="rbf")
k3 = GPy.kern.RBF(1, variance=1., lengthscale=0.1, name="rbf")

K1xx = k1.K(X,X) 
K2xx = k2.K(X,X) 

gp1 = GPy.models.GPRegression(X, Z, k1)
#mean1, Cov1 = gp1.predict_noiseless(XYnew, full_cov=True)

gp2 = GPy.models.GPRegression(X, Z, k2)
#mean2, Cov2 = gp2.predict_noiseless(XYnew, full_cov=True)

gp3 = GPy.models.GPRegression(X, Z, k3)
#mean3, Cov3 = gp3.predict_noiseless(XYnew, full_cov=True)

gp1.plot(title = 'lengthscale = 0.8')
gp2.plot(title = 'lengthscale = 0.5')
gp3.plot(title = 'lengthscale = 0.1')


# In[1]:


import numpy as np


# In[18]:


a = np.random.uniform(low=[-1,-10,20], high=[1,10,3], size = (10,3))


# In[20]:


a.shape[1]


# ## Plots para los experimentos de la reducción de objetivos

# In[59]:


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    return np.square(y - (5.1/(4*np.square(np.pi)))*np.square(x) + (5/np.pi)*x- 6) + 10*(1-(1./(8*np.pi)))*np.cos(x) + 10

def levy(x,y):
    x = scale_from_unity(x, -5, 10)
    y = scale_from_unity(y, 0, 15)
    w1 = (x-1)/4 + 1
    w2 = (y-1)/4 + 1
    return np.square(np.sin(np.pi*w1)) + (w2-1)**2*(1 + np.sin(2* np.pi*w2)**2)

scale = lambda x, a, b, a2, b2: (x-a)*(b2-a2)/(b-a)+a2
def scale_from_unity(x, a, b):
    return scale(x, -1,1,a,b)



# In[17]:


nsamples = 100
x = np.linspace(-1, 1, nsamples)
y = np.linspace(-1, 1, nsamples)
X, Y = np.meshgrid(x, y)
XY = np.stack((X.flatten(), Y.flatten()), axis = -1)
Z = griewank(X,Y)
Z2 = 30*(X**2 + Y**2)
Z3 = 30*(X**4 + Y**4)
Z4 = gramacy(X, Y)


# In[30]:


fig = plt.figure(figsize=(20,5))

plt.subplot(1, 4, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z, cmap='plasma')
ax.set_zlim(np.min(Z)-2,np.max(Z)+2)
plt.title("griewank")

plt.subplot(1, 4, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z2, cmap='plasma')
ax.set_zlim(np.min(Z2)-2,np.max(Z2)+2)
plt.title("30(x^2 + y^2)")

plt.subplot(1, 4, 3, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z3, cmap='plasma')
ax.set_zlim(np.min(Z3)-2,np.max(Z3)+2)
plt.title("30(x^4 + y^4)")

plt.subplot(1, 4, 4, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z4, cmap='plasma')
ax.set_zlim(np.min(Z4)-2,np.max(Z4)+2)
plt.title("gramacy")

plt.savefig('many.png', format='png', dpi=1200)


# In[52]:


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

nsamples = 50
x = np.linspace(-1, 1, nsamples)
y = np.linspace(-1, 1, nsamples)
X, Y = np.meshgrid(x, y)
XY = np.stack((X.flatten(), Y.flatten()), axis = -1)
Z = michalewicz(50, X,Y)
Z2 = michalewicz(100,X,Y)
Z3 = beale(X,Y)
Z4 = stybilinski(X, Y)

fig = plt.figure(figsize=(20,5))

plt.subplot(1, 4, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z, cmap='plasma')
ax.set_zlim(np.min(Z)-0.5,np.max(Z)+0.5)
#ax.view_init( elev = -2)
plt.title("michalewicz m=50")

plt.subplot(1, 4, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z2, cmap='plasma')
ax.set_zlim(np.min(Z2)-0.5,np.max(Z2)+0.5)
plt.title("michalewicz m=100")

plt.subplot(1, 4, 3, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z3, cmap='plasma')
ax.set_zlim(np.min(Z3)-0.5,np.max(Z3)+0.5)
plt.title("beale")

plt.subplot(1, 4, 4, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z4, cmap='plasma')
ax.set_zlim(np.min(Z4)-0.5,np.max(Z4)+0.5)
plt.title("stybilinski")

plt.savefig('many3.svg', format='svg', dpi=1200)


# In[63]:



nsamples = 500
x = np.linspace(-5, 10, nsamples)
y = np.linspace(0, 15, nsamples)
X, Y = np.meshgrid(x, y)
XY = np.stack((X.flatten(), Y.flatten()), axis = -1)
Z = branin( X,Y)
Z2 = 3*branin(X,Y)
Z3 = -1*branin(X, Y)
fig = plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z, cmap='plasma')
ax.set_zlim(np.min(Z)-0.5,np.max(Z)+0.5)
#ax.view_init( elev = -2)
plt.title("branin")

plt.subplot(1, 3, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z2, cmap='plasma')
ax.set_zlim(np.min(Z2)-0.5,np.max(Z2)+0.5)
plt.title("3 * branin")

plt.subplot(1, 3, 3, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y,Z3, cmap='plasma')
ax.set_zlim(np.min(Z3)-0.5,np.max(Z3)+0.5)
plt.title("- branin")
plt.savefig('branin.svg', format='svg', dpi=1200)


# In[48]:





# In[55]:


x = np.array([177.31,71.04,3, 77.3,3,131, 228.12, 300, 100,3, 194.67, 159.22,3, 41.59, 65.12, 24.43,10.52,135.54,35.60,111.65,76.41,6.25, 70.43,47.86,3,262.4,222.53,289.7,42.61,189.26])
y = np.array([84.37,3, 40, 56.45, 16, 3, 43.35, 3, 9.6,74.86, 3, 19.16,3,3.12, 3.85, 8.6, 49.43,100,15.32, 3,3.41, 3, 37.93, 81.94,100,3.15,3,7.07, 94.75,16.4])
XY = np.array(list(zip(x,y)))
Z1 = np.array([2.564915, 0.138882, 0.546864, 0.793237, 0.284178, 0.148019, 0.660461,0.211924,0.239453,1.084572,0.143567,0.327189,0.136217,0.15222, 0.153011, 0.214185, 0.651027,1.4062, 0.268085, 0.142576, 0.140056, 0.140528, 0.603364, 1.1367, 1.3363,0.141566, 0.139783, 0.197364, 1.270239, 0.295642])
Z2 = np.array([10924148, 84080, 191024, 1605076,86840,198800,9046704,835664,454164,349216,386736,1975824,22696, 49648, 96588,73880, 0.255928, 7888928, 162256, 156240,91440, 24744, 936904, 1153376, 459344, 653568,488656, 2158704,1158456, 2296388])
Z1_= Z1.reshape(Z1.size, 1)
Z2_= Z2.reshape(Z2.size, 1)


# In[81]:


npredict = 25
xnew = np.linspace(3, 300, npredict)
ynew = np.linspace(3, 100, npredict)
Xnew, Ynew = np.meshgrid(xnew, ynew)
XYnew = np.stack((Xnew.flatten(), Ynew.flatten()), axis = -1)

k1 = GPy.kern.sde_Matern52(2, variance=1, lengthscale=None, ARD=False, active_dims=None, name='Mat52')
gp1 = GPy.models.GPRegression(XY, Z1_, k1)
m1, v1 = gp1.predict(XYnew, full_cov = True)

k2 = GPy.kern.sde_Matern52(2, variance=.01, lengthscale=None, ARD=False, active_dims=None, name='Mat52')
gp2 = GPy.models.GPRegression(XY, Z2_, k2)
m2, v2 = gp2.predict(XYnew, full_cov = True)

fig = plt.figure(figsize=(10,5))

# GP plots

#plt.subplot(2, 3, 4, projection='3d')
plt.subplot(1, 2, 1, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m1.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m1)-0.01,np.max(m1)+0.01)
plt.title("Running time")

#plt.subplot(2, 3, 5, projection='3d')
plt.subplot(1, 2, 2, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(Xnew, Ynew, m2.reshape(npredict, npredict), cmap='plasma')
ax.set_zlim(np.min(m2)-0.01,np.max(m2)+0.01)
plt.title("Model size")


# In[ ]:





# In[49]:


Z1_.shape


# In[ ]:




