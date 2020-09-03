#%% prepare the data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_dir = 'data/abalone/abalone.data'
labels = ['Male','Female','Infant'] # male, Femail, Infant

# Read data from file
with open(data_dir) as f:
    lines = f.readlines()

samples = []

for line in lines:
    # features = line.strip().split(',')
    # samples.append(features[0:3])
    samples.append(line.strip().split(','))

# data = pd.DataFrame(samples,columns=['Sex','Length','Diameter'])


data = pd.DataFrame(samples,columns=['Sex','Length','Diameter','Height', \
    'Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings'])

#%% Scatter plot

# print(max(data['Diameter']))

sns.pairplot(data,hue='Sex')
# sns.distplot(data['Length'])
plt.show()


