#!/usr/bin/env python
# coding: utf-8

# ## 1) import libraries
# ## 2) make your training data(random data in this example)
# ## 3) make targets(the correct values)
# ## 4) plot the train data
# ## 5) create weights - By PC
# ## 6) create biases - By PC
# ## 7) set a learning rate - By PC

# In[ ]:





# # 1) Libraries

# In[1]:


# numpy contains all the mathematical operations, it is very fast
import numpy as np

# mathplotlib provides a nice interface and plot your data in eny type of plot
import matplotlib.pyplot as plt

# mpl_toolkits plot 3D graphs 
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:





# # 2) Generate random data to train on

# In[2]:


# in real life we import data from real data sources 
# in this particular example we want to generate data manually

# number of data
observations = 1000

# generate data randomly using numpy
xs = np.random.uniform(low=-10,high=10,size=(observations,1))
xz = np.random.uniform(low=-10,high=10,size=(observations,1))

# create a observation * 2 matrix from xs and xz
inputs = np.column_stack((xs,xz))

# To be sure of data shape
print(inputs.shape)


# In[ ]:





# # 3) make targets(the correct values)
# # ----------------------------------------------------
# ## targets = f(x,z) = 2x - 3z + 5 + noise
# ### Weights --> W1 = 2 and W2 = -3
# ### biase = 5
# ### why noise? real data always contains noise

# In[4]:


# create noise using numpy
noise = np.random.uniform(-1,1,(observations,1))

# create target: 
targets = 2*xs - 3*xz + 5 + noise

# check target shape
print(targets.shape)


# In[ ]:





# # 4) plot the train data

# In[7]:


# casting 1000*1 array to 1000*1 using numpy
targets = targets.reshape(observations,)

# make a plot platform
fig = plt.figure(figsize=(10,10))
# add a subplot and its type = 3D
ax = fig.add_subplot(111,projection = '3d')
# plot data
ax.plot(xs,xz,targets)

# set lables
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')

ax.view_init(azim=100)

# show plot
plt.show()

targets = targets.reshape(observations,1)


# In[ ]:





# # initialize variables

# In[8]:


init_range = 0.1

weights = np.random.uniform(-init_range,init_range,size=(2,1)) # we have 2*1 array for two args

biases = np.random.uniform(-init_range,init_range,size=1) # we have one bias

print(weights)
print(biases)


# In[ ]:





# # Set a learning rate

# In[9]:


# learning_rate value is experimental
learning_rate = 0.02


# In[ ]:





# # Train the model

# In[10]:


for i in range(100):
    # matrix operation of numpy
    outputs = np.dot(inputs,weights) + biases
    
    deltas = outputs - targets
    
    # calculate the loss(L2-norm loss) 
    # formula --> Sigmaof(yi - ti)^2 on any i
    loss = np.sum(deltas ** 2) / 2 / observations
    
    print(loss)

    deltas_scaled = deltas / observations

    # update weights
    weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
    
    # update biases
    biases = biases - learning_rate * np.sum(deltas_scaled)


# In[ ]:





# # Print optimized weights and biases if we have good results

# In[11]:


print('W = ', weights)
print('B = ', biases)


# In[ ]:





# # plot result against targets

# In[12]:


plt.plot(outputs,targets)


# In[ ]:




