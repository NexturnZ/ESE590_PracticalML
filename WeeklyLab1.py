#%% prepare the data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_dir = 'data/abalone/abalone.data'
labels = ['Male','Female','Infant'] # male, Femail, Infant

# initialize list of features respectively
_Sex = []
_Length = []
_Diameter = []
_Height = []
_Whole_weight = []
_Shucked_weight = []
_Viscera_weight = []
_Shell_weight = []
_Rings = []

# Read data from file
with open(data_dir) as f:
    lines = f.readlines()

# extract features
for line in lines:
    features = line.strip().split(',')
    _Sex.append(features[0])
    _Length.append(float(features[1]))
    _Diameter.append(float(features[2]))
    _Height.append(float(features[3]))
    _Whole_weight.append(float(features[4]))
    _Shucked_weight.append(float(features[5]))
    _Viscera_weight.append(float(features[6]))
    _Shell_weight.append(float(features[7]))
    _Rings.append(int(features[8]))

# observe the range of value of each feature
print('\tlength\tdiameter\theight\twhole\tshucked\tviscera\tshell\trings')
print('min\t%.3f\t%.3f\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t' \
        %(min(_Length),min(_Diameter),min(_Height),min(_Whole_weight), \
            min(_Shucked_weight),min(_Viscera_weight),min(_Shell_weight),min(_Rings)))
print('max\t%.3f\t%.3f\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t' \
        %(max(_Length),max(_Diameter),max(_Height),max(_Whole_weight), \
            max(_Shucked_weight),max(_Viscera_weight),max(_Shell_weight),max(_Rings)))

# classify data

# classify Sex
temp_m = [_Sex[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Sex[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Sex[i] for i in range(len(lines)) if _Sex[i]=='I']
Sex = dict(zip(labels,[temp_m,temp_f,temp_i]))

# classify length
temp_m = [_Length[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Length[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Length[i] for i in range(len(lines)) if _Sex[i]=='I']
Length = dict(zip(labels,[temp_m,temp_f,temp_i]))

# classify diameter
temp_m = [_Diameter[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Diameter[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Diameter[i] for i in range(len(lines)) if _Sex[i]=='I']
Diameter = dict(zip(labels,[temp_m,temp_f,temp_i]))

# classify height
temp_m = [_Height[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Height[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Height[i] for i in range(len(lines)) if _Sex[i]=='I']
Height = dict(zip(labels,[temp_m,temp_f,temp_i]))

# classify whole_weight
temp_m = [_Whole_weight[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Whole_weight[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Whole_weight[i] for i in range(len(lines)) if _Sex[i]=='I']
Whole_weight = dict(zip(labels,[temp_m,temp_f,temp_i]))

# classify Shucked_weight
temp_m = [_Shucked_weight[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Shucked_weight[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Shucked_weight[i] for i in range(len(lines)) if _Sex[i]=='I']
Shucked_weight = dict(zip(labels,[temp_m,temp_f,temp_i]))

# classify Viscera_weight
temp_m = [_Viscera_weight[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Viscera_weight[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Viscera_weight[i] for i in range(len(lines)) if _Sex[i]=='I']
Viscera_weight = dict(zip(labels,[temp_m,temp_f,temp_i]))

# classify Rings
temp_m = [_Rings[i] for i in range(len(lines)) if _Sex[i]=='M']
temp_f = [_Rings[i] for i in range(len(lines)) if _Sex[i]=='F']
temp_i = [_Rings[i] for i in range(len(lines)) if _Sex[i]=='I']
Rings = dict(zip(labels,[temp_m,temp_f,temp_i]))

#%% TODO Scatter Plots at three dimension: length, diameter, height
length_major = plt.MultipleLocator(0.1)
diameter_major = plt.MultipleLocator(0.1)
height_major = plt.MultipleLocator(0.1)

# pdf of length
sub1 = plt.subplot(3,3,1,xlabel='length',ylabel='length')
sns.kdeplot(Length['Male'])
sns.kdeplot(Length['Female'])
sns.kdeplot(Length['Infant'])

# length v.s. diameter
sub2 = plt.subplot(3,3,4,xlabel='Length',ylabel='diameter')
sns.scatterplot(Length['Male'],Diameter['Male'],s=5)
sns.scatterplot(Length['Female'],Diameter['Female'],s=5)
sns.scatterplot(Length['Infant'],Diameter['Infant'],s=5)
sub2.xaxis.set_major_locator(length_major)
sub2.yaxis.set_major_locator(diameter_major)

# length v.s. height
sub3 = plt.subplot(3,3,7,xlabel='Length',ylabel='height')
sub3.scatter(Length['Male'],Height['Male'],s=0.5,c='r')
sub3.scatter(Length['Female'],Height['Female'],s=0.5,c='g')
sub3.scatter(Length['Infant'],Height['Infant'],s=0.5,c='b')
sub3.xaxis.set_major_locator(length_major)
sub3.yaxis.set_major_locator(height_major)

# diameter v.s.
sub4 = plt.subplot(3,3,3,xlabel='Length',ylabel='height')
sub4.scatter(Length['Male'],Height['Male'],s=0.5,c='r')
sub4.scatter(Length['Female'],Height['Female'],s=0.5,c='g')
sub4.scatter(Length['Infant'],Height['Infant'],s=0.5,c='b')
sub4.xaxis.set_major_locator(length_major)
sub4.yaxis.set_major_locator(height_major)

# TODO sub5

plt.show()


#%% TODO Hexbin plots

#%% TODO Density Plots

#%% TODO Strip Plots

#%% TODO Swarm Plots

#%% TODO Box Plots

#%% TODO Bar Plots







