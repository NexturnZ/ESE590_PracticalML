import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data_dir = 'data/scale/balance-scale.data'
label_list = ['L','B','R'] # three states of the scale

purity_threshold = 0.85

class Node:
    def __init__(self, root, leaf,dataset):
        self.root = root
        self.feature = None
        self.cond_feature = []
        self.Left_sub = None
        self.Middle_sub = None
        self.Right_sub = None
        self.leaf = leaf
        self.dataset = dataset

    # TODO
    def forward(self, attri_val):
        pass

def pmf(val,conditions=None):
    num = len(val[0,:])
    dim = 5*np.ones(num,dtype=int)
    dim = dim.tolist()
    pmf = np.zeros(shape=dim,dtype=int)

    idx_map = (np.array(list(range(np.prod(dim))))+1).reshape(dim)
    for i1 in range(len(val)):
        sample = val[i1,:]
        # sample.astype(np.float64)
        idx = np.dot(np.flip(np.logspace(0,3,4,endpoint=True,base=5)),sample-1)+1
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

def leaf_judgement(subset, labels):
    Num = len(subset)
    L = 0
    B = 0
    R = 0

    if Num != 0:
        for i1 in subset:
            if labels[i1] == 'L':
                L +=1
            elif labels[i1] == 'B':
                B +=1
            else:
                R +=1
        
        purity = [L/Num, B/Num, R/Num]

        if np.max(purity) > purity_threshold:
            leaf_state = label_list[np.argmax(purity)]
        else:
            leaf_state = 'not leaf'
    else:
        # leaf_state = 'leaf'
        leaf_state = 'B'
    
    return leaf_state


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
        middle_set = []
        right_set = []
        subset = train_data[node.dataset]
        for i1 in range(len(subset)):
            if subset[i1,node.feature]<thresholds[node.feature]:
                left_set.append(node.dataset[i1])
            elif subset[i1,node.feature] == thresholds[node.feature]:
                middle_set.append(node.dataset[i1])
            else:
                right_set.append(node.dataset[i1])
        
        left_class = leaf_judgement(left_set,train_labels)
        node.Left_sub = Node(root=node,leaf=left_class,dataset=left_set)
        node.Left_sub.cond_feature = node.cond_feature.copy()
        node.Left_sub.cond_feature.append(node.feature)

        # if all features are asked, this node is a leaf node
        node.Left_sub.cond_feature.sort()
        if node.Left_sub.leaf == 'not leaf' and node.Left_sub.cond_feature == features:
            # node.Left_sub.leaf = 'leaf'
            node.Left_sub.leaf = 'B'

        if node.Left_sub.leaf == 'not leaf':
            node_queue.append(node.Left_sub)
            left_pmf = pmf(train_data[left_set])
            Left_feature = feature_select(joint_pmf=left_pmf,conditions=node.Left_sub.cond_feature)
            node.Left_sub.feature = Left_feature



        middle_class = leaf_judgement(middle_set,train_labels)
        node.Middle_sub = Node(root=node,leaf=middle_class,dataset=middle_set)
        node.Middle_sub.cond_feature = node.cond_feature.copy()
        node.Middle_sub.cond_feature.append(node.feature)

        # if all features are asked, this node is a leaf node
        node.Middle_sub.cond_feature.sort()
        if node.Middle_sub.leaf == 'not leaf' and node.Middle_sub.cond_feature == features:
            # node.Middle_sub.leaf = 'leaf'
            node.Middle_sub.leaf = 'B'

        if node.Middle_sub.leaf == 'not leaf':
            node_queue.append(node.Middle_sub)
            left_pmf = pmf(train_data[left_set])
            Left_feature = feature_select(joint_pmf=left_pmf,conditions=node.Middle_sub.cond_feature)
            node.Middle_sub.feature = Left_feature



        right_class = leaf_judgement(right_set,train_labels)
        node.Right_sub = Node(root=node,leaf=right_class,dataset=right_set)
        node.Right_sub.cond_feature = node.cond_feature.copy()
        node.Right_sub.cond_feature.append(node.feature)

        # if all features are asked, this node is a leaf node
        node.Right_sub.cond_feature.sort()
        if node.Right_sub.leaf == 'not leaf' and node.Right_sub.cond_feature == features:
            # node.Right_sub.leaf = 'leaf'
            node.Right_sub.leaf = 'B'

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
        if sample[idx] <= thresholds[idx]:
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
    raw_data = []
    raw_labels = []
    for line in lines:
        features = line.strip().split(',')
        for i in range(4):
            features[i+1] = float(features[i+1])
        raw_labels.append(features[0])
        raw_data.append(features[1:])

    ag_data = raw_data+raw_data
    ag_labels = raw_labels+raw_labels

    data = np.array(ag_data) # [Left_weight, Left_distance, Right_weight, Right_distance]
    labels = np.array(ag_labels)

    state = np.random.get_state()
    np.random.shuffle(data) # shuffle data
    np.random.set_state(state)
    np.random.shuffle(labels)
    
    # split data into training set & test set
    train_data = data[:1100]
    train_data = np.array(train_data)
    train_labels = labels[:1100]

    test_data = data[1100:]
    test_data = np.array(test_data)
    test_labels = labels[1100:]

    # use 3 as thresholds for every features
    thresholds = 3*np.ones(len(train_data[0]))

    root = train(train_data,train_labels,thresholds)
    acc = test(test_data,test_labels,root,thresholds)
    print('accuracy is %f'%acc)

    # accuracy = test(test_data,test_labels,root)

    # data_plot = pd.DataFrame(data,columns=['Lweight','Ldistance','Rweight','Rdistance'])
    # plt.figure()
    # sns.distplot(data_plot['Rdistance'])
    # plt.show()

if __name__ == "__main__":
    main()
