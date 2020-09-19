import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data_dir = 'data/scale/balance-scale.data'
labels = ['Left','Right','Balanced'] # three states of the scale


states = [1,2,3,4,5]


def entropy(val,condition=None):
    pass

def main():
    pass

def pdf(val):
    num = len(val)
    dim = np.ones(num,dtype=int)
    dim = 5*dim.tolist()
    pdf = np.zeros(shape=dim,dtype=int)
    # TODO
    for i1 in range(len(val)):
        pdf[val[i1]-1] += 1
    return pdf

if __name__ == "__main__":
    with open(data_dir) as f:
        lines = f.readlines()

    # extract features
    samples = []
    for line in lines:
        features = line.strip().split(',')
        for i in range(4):
            features[i+1] = int(features[i+1])
        samples.append(features)


    data = pd.DataFrame(samples,columns=['state','Lweight','Ldistance','Rweight','Rdistance'])

    Lweight_pdf = pdf(data['Lweight'])
    plt.figure()
    sns.distplot(data['Rdistance'])
    plt.show()

