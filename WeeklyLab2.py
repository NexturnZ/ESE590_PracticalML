import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter

# use CAR data set
data_dir = 'data/car/car.data'
label_list = ['unacc','acc','good','vgood'] # unacceptable, acceptable, good, very-good
feature_map = [['vhigh','high','med','low'], \
                ['vhigh','high','med','low'], \
                ['2','3','4','5more'], \
                ['2','4','more'], \
                ['small','med','big'], \
                ['low','med','high']]
thresholds = [2,2,2,2,2,2] # threshold for each feature

# # use scale data set
# data_dir = 'data/scale/balance-scale.data'
# label_list = ['L','B','R']
# feature_map = [['1','2','3','4','5'], \
#                 ['1','2','3','4','5'], \
#                 ['1','2','3','4','5'], \
#                 ['1','2','3','4','5']]
# thresholds = [3,3,3,3] # threshold for each feature

# # use beast-cancer data set
# data_dir = 'data/breast-cancer/breast-cancer.data'
# label_list = ['no-recurrence-events','recurrence-events']
# feature_map = [['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99'], \
#                 ['lt40','ge40','premeno'], \
#                 ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44', \
#                  '45-49','50-54','55-59'], \
#                 ['0-2','3-5','6-8','9-11','12-14','15-17','18-20','21-23','24-26', \
#                  '27-29','30-32','33-35','36-39'], \
#                 ['yes','no'], \
#                 ['1', '2', '3'], \
#                 ['left', 'right'], \
#                 ['left_up', 'left_low', 'right_up',	'right_low', 'central'], \
#                 ['yes', 'no']]
# thresholds = [1,4,1,6,7,1,1,1,3,1]


purity_threshold = 0.85



class Node:
    def __init__(self, root, leaf,dataset):
        self.root = root
        self.feature = None
        self.cond_feature = []
        self.Left_sub = None
        # self.Middle_sub = None
        self.Right_sub = None
        self.leaf = leaf
        self.dataset = dataset

def pmf(val,conditions=None):
    num = len(val[0,:]) # number of dimensions
    dim = []   # initialize dimension
    for i1 in range(num):
        dim.append(len(feature_map[i1]))


    pmf = np.zeros(shape=dim,dtype=int)

    _tmp = np.zeros(num)
    for i1 in range(num-1):
        _tmp[i1] = np.prod(dim[i1+1:])
    _tmp[-1] = 1


    idx_map = (np.array(list(range(np.prod(dim))))).reshape(dim)
    for i1 in range(len(val)):
        sample = val[i1,:]
        _feature = np.zeros(num)
        for i1 in range(num):
            _feature[i1] = feature_map[i1].index(sample[i1])
        
        idx = np.dot(_tmp,_feature)
        pmf[idx_map==idx] += 1

    pmf = pmf/len(val)

    return pmf

# joint_pmf: joint pmf (dimension = [feature, conditions])
# feature: the feature to calculate entropy for
def entropy(joint_pmf,conditions=None):
    if conditions==None:
        joint_pmf[joint_pmf==0] = 0.00001
        entro = -np.sum(joint_pmf*np.log2(joint_pmf))
    else:
        # H(X|Y) = H(X,Y)-H(Y)
        joint_entro = entropy(joint_pmf) # calculate

        dim = list(range(len(joint_pmf.shape)))
        for i1 in conditions:
            dim.remove(i1)
        condVal_pmf = np.sum(joint_pmf,axis=tuple(dim))
        condVal_entro = entropy(condVal_pmf)
        entro = joint_entro-condVal_entro
    return entro

def feature_select(joint_pmf,conditions=None):
    # extract remaining feature index
    dim = list(range(len(joint_pmf.shape)))
    if conditions != None:
        for i1 in conditions:
            dim.remove(i1)


    entropies = np.zeros(len(joint_pmf.shape))
    for i1 in dim:
        _features = dim.copy()
        _features.remove(i1)
        margin_pmf = np.sum(joint_pmf,axis=tuple(_features))
        if conditions != None:
            _conditions = conditions_mapping(conditions=conditions,extracted_feature=_features)
        else: 
            _conditions = None
        
        _entropy = entropy(joint_pmf=margin_pmf,conditions=_conditions)
        entropies[i1] = _entropy

    feature = np.argmax(entropies)
    return feature

def leaf_judgement(subset, labels, feature):
    Num = len(subset)
    if Num != 0:
        purities = purity_cal(subset, labels, feature)
        if np.max(purities) > purity_threshold:
            leaf_state = label_list[np.argmax(purities)]
        else:
            leaf_state = 'not leaf'
    else:
        leaf_state = 'leaf'
    
    return leaf_state

def purity_cal(subset, labels, feature):
    Num = len(subset)
    purities = np.zeros(len(label_list))

    for i1 in subset:
        for i2 in range(len(label_list)):
            if labels[i1] == label_list[i2]:
                purities[i2] +=1

    final_purities = purities/Num
    return final_purities

def conditions_mapping(conditions,extracted_feature):
    _conditions = np.array(conditions)
    _ext = np.array(extracted_feature)
    for i1 in range(len(conditions)):
        _conditions[i1] -= len(_ext[_conditions[i1]>_ext])
    
    conditions_new = _conditions.tolist()
    return conditions_new

def train(train_data,train_labels,thresholds):

    
    features = list(range(len(train_data[0]))) # index of features
    joint_pmf = pmf(train_data)

    # extract feature with larges entropy
    feature = feature_select(joint_pmf)

    # build root node
    root_set = np.array(list(range(len(train_data))))
    root = Node(root=None,leaf='not leaf',dataset=root_set)
    root.feature = feature

    node_queue = [root]

    while node_queue:
        node = node_queue[0]
        left_set = []
        # middle_set = []
        right_set = []
        subset = train_data[node.dataset]
        for i1 in range(len(subset)):
            if feature_map[node.feature].index(subset[i1,node.feature])<thresholds[node.feature]:
                left_set.append(node.dataset[i1])
            # elif feature_map[node.feature].index(subset[i1,node.feature])==thresholds[node.feature]:
            #     middle_set.append(node.dataset[i1])
            else:
                right_set.append(node.dataset[i1])
        
        ####################### construct left offspring node
        left_class = leaf_judgement(left_set,train_labels,node.feature)
        node.Left_sub = Node(root=node,leaf=left_class,dataset=left_set)
        node.Left_sub.cond_feature = node.cond_feature.copy()
        node.Left_sub.cond_feature.append(node.feature)

        # if all features are asked, this node is a leaf node
        node.Left_sub.cond_feature.sort()
        if node.Left_sub.leaf == 'not leaf' and node.Left_sub.cond_feature == features:
            _purity = purity_cal(left_set,train_labels,node.feature)
            node.Left_sub.leaf = label_list[np.argmax(_purity)]
            # node.Left_sub.leaf = 'leaf'

        if node.Left_sub.leaf == 'not leaf':
            node_queue.append(node.Left_sub)
            left_pmf = pmf(train_data[left_set])
            Left_feature = feature_select(joint_pmf=left_pmf,conditions=node.Left_sub.cond_feature)
            node.Left_sub.feature = Left_feature

        # ####################### construct middle offspring node
        # middle_class = leaf_judgement(middle_set,train_labels,node.feature)
        # node.Middle_sub = Node(root=node,leaf=middle_class,dataset=middle_set)
        # node.Middle_sub.cond_feature = node.cond_feature.copy()
        # node.Middle_sub.cond_feature.append(node.feature)

        # # if all features are asked, this node is a leaf node
        # node.Middle_sub.cond_feature.sort()
        # if node.Middle_sub.leaf == 'not leaf' and node.Middle_sub.cond_feature == features:
        #     _purity = purity_cal(middle_set,train_labels,node.feature)
        #     node.Middle_sub.leaf = label_list[np.argmax(_purity)]
        #     # node.Left_sub.leaf = 'leaf'

        # if node.Middle_sub.leaf == 'not leaf':
        #     node_queue.append(node.Middle_sub)
        #     middle_pmf = pmf(train_data[middle_set])
        #     Middle_feature = feature_select(joint_pmf=middle_pmf,conditions=node.Middle_sub.cond_feature)
        #     node.Middle_sub.feature = Middle_feature

        ########################### construct right offspring node
        right_class = leaf_judgement(right_set,train_labels,node.feature)
        node.Right_sub = Node(root=node,leaf=right_class,dataset=right_set)
        node.Right_sub.cond_feature = node.cond_feature.copy()
        node.Right_sub.cond_feature.append(node.feature)

        # if all features are asked, this node is a leaf node
        node.Right_sub.cond_feature.sort()
        if node.Right_sub.leaf == 'not leaf' and node.Right_sub.cond_feature == features:
            _purity = purity_cal(right_set,train_labels,node.feature)
            node.Right_sub.leaf = label_list[np.argmax(_purity)]
            # node.Right_sub.leaf = 'leaf'

        if node.Right_sub.leaf == 'not leaf':
            node_queue.append(node.Right_sub)
            right_pmf = pmf(train_data[right_set])
            Right_feature = feature_select(joint_pmf=right_pmf,conditions=node.Right_sub.cond_feature)
            node.Right_sub.feature = Right_feature
        
        node_queue.pop(0) # remove first node

    return root

def test(test_data,test_labels,root,thresholds):
    correct = 0
    results = []
    Num = len(test_data)
    for i1 in range(Num):
        result = forward(test_data[i1],root,thresholds)
        results.append(result)
        if result == test_labels[i1]:
            correct += 1

    acc = correct/Num
    # print('\n')
    # print(Counter(results))

    return acc

def forward(sample,root,thresholds):
    node = root
    while node.leaf == 'not leaf':
        idx = node.feature
        if feature_map[idx].index(sample[idx]) < thresholds[idx]:
            node = node.Left_sub
        # elif feature_map[idx].index(sample[idx]) == thresholds[idx]:
        #     node = node.Middle_sub
        else:
            node = node.Right_sub

    result = node.leaf
    return result

def main():
    # load data from file
    with open(data_dir) as f:
        lines = f.readlines()

    # extract features
    data = []
    labels = []
    data_plot = []
    for line in lines:
        features = line.strip().split(',')
        if '?' not in features:
            # labels.append(features[0])
            # data.append(features[1:])
            data_plot.append(features)
            labels.append(features[-1])
            data.append(features[:-1])


    data_plot = pd.DataFrame(data_plot,columns=['buying','maint','doors','persons','lug','safetys','label'])

    data = np.array(data) 
    labels = np.array(labels)

    state = np.random.get_state()
    np.random.shuffle(data) # shuffle data
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    # split data into training set & test set
    _tmp = int(np.floor(len(labels)/3*2))
    train_data = data[:_tmp]
    train_labels = labels[:_tmp]

    test_data = data[_tmp:]
    test_labels = labels[_tmp:]


    root = train(train_data,train_labels,thresholds)
    acc = test(test_data,test_labels,root,thresholds)
    print('\naccuracy is %f\n'%acc)


    # plt.figure()
    # # sns.pairplot(data_plot,hue='label',vars=['buying','maint','safety'])
    # sns.barplot(hue='label',x='buying',y='safety',data=data)
    # plt.show()

if __name__ == "__main__":
    main()
