import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data_dir = 'data/car/car.data'
label_list = ['unacc','acc','good','vgood'] # unacceptable, acceptable, good, very-good
feature_list = ['buying','maint','doors','persons','lug-boot','safety']
feature_map = [['vhigh','high','med','low'], \
                ['vhigh','high','med','low'], \
                ['2','3','4','5more'], \
                ['2','4','more'], \
                ['small','med','big'], \
                ['low','med','high']]

purity_threshold = 0.85

# use 3 as thresholds for every features
thresholds = [2,2,2,2,2,2]

class Node:
    def __init__(self, root, leaf,dataset):
        self.root = root
        self.feature = None
        self.cond_feature = []
        self.Left_sub = None
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

    # index of features [[Left_weight, Left_distance, Right_weight, Right_distance]
    features = list(range(len(train_data[0])))
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
        right_set = []
        subset = train_data[node.dataset]
        for i1 in range(len(subset)):
            if feature_map[node.feature].index(subset[i1,node.feature])<thresholds[node.feature]:
                left_set.append(node.dataset[i1])
            else:
                right_set.append(node.dataset[i1])
        
        # construct left offspring node
        left_class = leaf_judgement(left_set,train_labels,node.feature)
        node.Left_sub = Node(root=node,leaf=left_class,dataset=left_set)
        node.Left_sub.cond_feature = node.cond_feature.copy()
        node.Left_sub.cond_feature.append(node.feature)

        # if all features are asked, this node is a leaf node
        node.Left_sub.cond_feature.sort()
        if node.Left_sub.leaf == 'not leaf' and node.Left_sub.cond_feature == features:
            _purity = purity_cal(left_set,train_labels,node.feature)
            node.Left_sub.leaf = label_list[np.argmax(_purity)]

        if node.Left_sub.leaf == 'not leaf':
            node_queue.append(node.Left_sub)
            left_pmf = pmf(train_data[left_set])
            Left_feature = feature_select(joint_pmf=left_pmf,conditions=node.Left_sub.cond_feature)
            node.Left_sub.feature = Left_feature


        # construct right offspring node
        right_class = leaf_judgement(right_set,train_labels,node.feature)
        node.Right_sub = Node(root=node,leaf=right_class,dataset=right_set)
        node.Right_sub.cond_feature = node.cond_feature.copy()
        node.Right_sub.cond_feature.append(node.feature)

        # if all features are asked, this node is a leaf node
        node.Right_sub.cond_feature.sort()
        if node.Right_sub.leaf == 'not leaf' and node.Right_sub.cond_feature == features:
            _purity = purity_cal(right_set,train_labels,node.feature)
            node.Right_sub.leaf = label_list[np.argmax(_purity)]

        if node.Right_sub.leaf == 'not leaf':
            node_queue.append(node.Right_sub)
            right_pmf = pmf(train_data[right_set])
            Right_feature = feature_select(joint_pmf=right_pmf,conditions=node.Right_sub.cond_feature)
            node.Right_sub.feature = Right_feature
        
        node_queue.pop(0) # remove first node

    return root

def test(test_data,test_labels,root,thresholds):
    results = 0
    Num = len(test_data)
    for i1 in range(Num):
        result = forward(test_data[i1],root,thresholds)
        if result == test_labels[i1]:
            results += 1

    acc = results/Num
    return acc

def forward(sample,root,thresholds):
    node = root
    while node.leaf == 'not leaf':
        idx = node.feature
        if feature_map[idx].index(sample[idx]) < thresholds[idx]:
            node = node.Left_sub
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
    for line in lines:
        features = line.strip().split(',')
        labels.append(features[-1])
        data.append(features[:6])

    data = np.array(data) # [Left_weight, Left_distance, Right_weight, Right_distance]
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


if __name__ == "__main__":
    main()
