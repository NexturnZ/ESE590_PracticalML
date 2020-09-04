#%% prepare the data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data_dir = 'data/abalone/abalone.data'
labels = ['Male','Female','Infant'] # male, Femail, Infant

# Read data from file
with open(data_dir) as f:
    lines = f.readlines()

# extract features
samples = []
for line in lines:
    features = line.strip().split(',')
    for i in range(7):
        features[i+1] = float(features[i+1])
    features[8] = int(features[8])
    samples.append(features)

data = pd.DataFrame(samples,columns=['Sex','Length','Diameter','Height', \
    'Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings'])

#%% Scatter plots for three dimension: length, diameter, height
sns.pairplot(data,hue='Sex',vars=['Length','Diameter','Height'])

#%% Hexbin plots for length vs. rings
hexbin = plt.figure()
plt.hexbin(data['Length'], data['Rings'], gridsize=(10,10), cmap='Purples')
plt.xlabel('Length')
plt.ylabel('Rings')

#%% Density Plots for Whole_weight
dist = plt.figure()
sns.distplot(data['Whole_weight'])

#%% Strip Plots for shucked weight
strip = plt.figure()
sns.stripplot(x='Sex', y='Shucked_weight', data=data, jitter=True)
plt.grid()

#%% Swarm Plots for viscera weight
swarm = plt.figure()
sns.swarmplot(x='Sex',y='Viscera_weight', data=data)
plt.grid()

#%% Box Plots for shell weight
box = plt.figure()
sns.boxplot(x='Sex',y='Shell_weight',data=data)
plt.grid()

#%% Bar Plots length vs. Rings
bar = plt.figure()
sns.barplot(x='Rings',y='Length',data=data)
plt.show()

# %%
