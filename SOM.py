import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.spatial import distance
from scipy.spatial import distance_matrix
import math
import random
import pandas as pd


#create empty list to store variables
input =[]

#read dataset and get the column length
df=pd.read_excel('AnimalData.xls', index_col=0)  
L_input=len(df.columns)

#initialize random weights for 20 by 20 grid
#weights=np.random.uniform(low=0, high=1, size=(L_input,20,20))

#formula for weights initilization
for i in range(0, L_input):
    firstPart = np.mean(df.iloc[:, i]) + np.random.uniform(-0.1, 0.1)
    secondPart = np.mean(df.iloc[:, i]) * np.random.uniform(low=0, high=1, size=(20, 20))
    weights_temp = firstPart * secondPart
    weights[i, :, :] = weights_temp
    
#define neighborhood function, given target node, neighborhodd size and grid size 
def find_neighbor_indices(i, j,neighborhoodSize,hiddenSize):
    # function finds the neighboring rows and columns to include
    # i : i-th index
    # j : j-th index
    # dist: how big the neighborhood should span
    
    rows = []
    columns = []
    
    # python indexing starts with 0 so adjust here
    i = i + 1
    j = j + 1
    rows = np.arange(i - int(neighborhoodSize), i + int(neighborhoodSize) + 1)
    columns = np.arange(j - int(neighborhoodSize), j + int(neighborhoodSize) + 1)
    
    # get neighbor indexes as a combination of rows and columns
    neighborhood = set()
    for row in rows:
        for column in columns:
    
            row = row % hiddenSize[0]
    
            column = column % hiddenSize[1]
    
            if row == 0:
                row = hiddenSize[0]
            if column == 0:
                column = hiddenSize[1]
    
            # do not update actual row, because it is used in the loop
            row_temp = row - 1
            column_temp = column - 1
    
            neighborhood.add((row_temp, column_temp))
    
    return neighborhood
                             

###### Preparation Work ######

# store each input pattern to a list
for index, rows in df.iterrows():
    # Create list for the current row
    input.append(rows)

### Start Training ###
epochs = 600

for epoch in range(epochs):
    
    coord=[]
    
    #implememt neighborhood size function
    neighbor_size=10*(1-(epoch/epochs))
    if neighbor_size > 1:
        neighbor_size = int(neighbor_size)
    else:
        neighbor_size=1
        
    #implement learning rate fuction    
    lr=0.9*(1-(epoch/epochs))
    if lr > 0.2:
        lr = lr
    else:
        lr=0.2
        
    #loop through each input pattern
    for r in range (0,len(input)):
        
        distance_all=[]
        neighbors=[]
        updated_weights_all=[]
    
        #calculate euclidean distance and find the winner node index
        for i in range(0,np.shape(weights)[1]):
            temp = []
            for j in range(0,np.shape(weights)[2]):
                s1 = input[r].values
                s2 = weights[:,i,j]
                d = distance.euclidean(s1, s2)
                temp.append(d)
            distance_all.append(temp)
        distance_all = np.asarray(distance_all)
        minpos = np.argmin(distance_all)

        #make 2d index
        two_dIndex = np.unravel_index(minpos, [20,20])
        x = two_dIndex[0]
        y = two_dIndex[1]
        
        #store winner node vector to a list
        target_index=[x,y]
        coord.append(target_index)
        
        #find all neighborhood
        neighborhood=find_neighbor_indices(x, y,neighbor_size,[20,20])

        #update weights for winner node and neighborhood nodes
        for j in neighborhood:
            update = (lr*(input[r].values-weights[:,j[0],j[1]]))
            weights[:,j[0],j[1]] =weights[:,j[0],j[1]] + update
    
    #pair label with each input pattern's winner node index    
    labels=df.index.values 
    T=dict(zip(labels, coord))
    
    ### Plot and Save map at every 100 epochs###     
    if epoch % 100 == 0:
        # repackage data into array-like for matplotlib, pythonically
        xs,ys = zip(*T.values())
        labels = T.keys()

        # display
        plt.figure(figsize=(10,8))
        plt.title('Scatter Plot', fontsize=20)
        plt.xlabel('x', fontsize=15)
        plt.ylabel('y', fontsize=15)
        plt.scatter(xs, ys, marker = 'o')
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(label, xy = (x, y),fontsize=20)
       
        plt.savefig(str(epoch)+'.png',bbox_inches = 'tight')
    
  

    
            
            
            
            
            
            
            
            
    
   
