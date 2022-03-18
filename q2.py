import numpy as np
import pandas as pd
from metrics import scores
from sklearn.model_selection import train_test_split

#load datasets
def load(filepath):
    data =  pd.read_csv(filepath, sep=" ", dtype=float,header=None)
    data = data.drop(22, axis=1)
    data = data.drop(23, axis=1)
    
    data_np = data.to_numpy()
    
    features = data_np[:, 0:21] 
    labels = data_np[:, 21:22] 
    return features, labels

train_features, train_labels = load("ann-train.data")
test_features, test_labels = load("ann-test.data")
train_labels = np.array(train_labels,dtype="int64")



#entropy calculation
def entropy(y):
    cnt = np.bincount(y)
    probs = cnt / len(y)
    ent = -np.sum([p * np.log(p) for p in probs if p > 0])
    return ent



class Node:
    def __init__(self, feature=None, threshold=None ,left_node=None, right_node=None, mark=None):
        self.feature = feature     
        self.threshold = threshold
        self.left_node = left_node       
        self.right_node = right_node       
        self.mark = mark           
    def is_leaf_node(self):
        return self.mark is not None
    
class DecisionTreeClassifier:
    def __init__(self,min_samples_split=2, max_depth=0, number_of_features=None):
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        self.number_of_features = number_of_features
        self.root = None        
        
    def fit(self, X, y):
        self.number_of_features = X.shape[1] if not self.number_of_features else min(self.number_of_features, X.shape[1])        
        self.root = self.build_node(X, y)
        
    def predict(self, X):
        array = []
        for x in X:
            array.append(self.bottom_up(x, self.root))
        return np.array(array)
    
    def bottom_up(self, x, node):
        if node.is_leaf_node():
            return node.mark
        if x[node.feature] <= node.threshold:  
            return self.bottom_up(x, node.left_node)
        else:
            return self.bottom_up(x, node.right_node)
    
    
    def build_node(self, X, y, depth=0):
        n_samples, n_features = X.shape         
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_mark = self.most_class(y)
            return Node(mark=leaf_mark)
        
        feature_idxs = np.random.choice(n_features, self.number_of_features, replace=False)        
        best_feature, best_thresh = self.best_split(X, y, feature_idxs)        
        left_idxs, right_idxs = self.split(X[:, best_feature], best_thresh)        
        left = self.build_node(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.build_node(X[right_idxs, :], y[right_idxs], depth+1)        
        return Node(best_feature, best_thresh,left, right)
        
        
    def best_split(self, X, y, feature_idxs):
        best_information_gain = -1
        split_idx, split_thresh = None, None
        for idx in feature_idxs:            
            X_column=X[:, idx]
            thresholds = np.unique(X_column)            
            for threshold in thresholds:                
                gain = self.information_gain(y, X_column, threshold)                
                if gain > best_information_gain:
                    best_information_gain = gain
                    split_idx = idx
                    split_thresh = threshold                    
        return split_idx, split_thresh
    
    
    def information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)        
        left_idxs, right_idxs = self.split(X_column, split_thresh)        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)        
        entropy_l, entropy_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n)*entropy_l + (n_r/n)*entropy_r        
        info_gain = parent_entropy - child_entropy        
        return info_gain
    
    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
        
    
    def most_class(self, y):
        most_class = np.argmax(np.bincount(y))
        return most_class
    
    def print_tree(self, node=None, depth=0):
        if not node:
            node = self.root
        if node.is_leaf_node():
            print('\t' * depth, "Leaf:", node.mark)
            return
        print('\t' * depth, "Split: X{} <= {} ".format(node.feature, node.threshold))
        self.print_tree(node.left_node, depth + 1)
        self.print_tree(node.right_node, depth + 1)
        


max_depth = [10, 15, 25, 30]
min_samples_split = [5, 10, 15]
hyper_parameters = {}
X_train, X_cv , y_train, y_cv = train_test_split(train_features, train_labels, test_size=0.2, random_state=35)
for max_depth_ in max_depth:
    for min_samples_split_ in min_samples_split:
        hyper_parameters["max_depth=" + str(max_depth_) + ",min_samples_split=" + str(min_samples_split_)] = []
        my_clf = DecisionTreeClassifier(max_depth=max_depth_, min_samples_split=min_samples_split_)
        model = my_clf.fit(X_train, y_train[:,0])
        cv_y_pred = my_clf.predict(X_cv)
        cm = scores(y_cv,cv_y_pred)
        micro_acc = 0
        for i in range(3):
            micro_acc += (cm[i,i]/np.sum(cm[i,:]) * np.sum(y_cv == i+1))
        micro_acc = micro_acc / y_cv.shape[0]
        hyper_parameters["max_depth=" + str(max_depth_) + ",min_samples_split=" + str(min_samples_split_)].append(micro_acc)
best_parameters = max(hyper_parameters, key=hyper_parameters.get)
print(f"Best hyperparameters chosen on cross validation set is {best_parameters}")
my_clf = DecisionTreeClassifier(max_depth=int(best_parameters.split(",")[0].split("=")[1]),min_samples_split = int(best_parameters.split(",")[1].split("=")[1]))
model = my_clf.fit(X_train, y_train[:,0])

print("\n")
print("PERFORMANCE ON TRAINING SET")
train_y_pred = my_clf.predict(train_features)
cm = scores(train_labels,train_y_pred)
print(cm)
print("Class based accuracies: ")
for i in range(3):
    print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))

print("\n")
print("PERFORMANCE ON TEST SET")
test_y_pred = my_clf.predict(test_features)
cm = scores(test_labels,test_y_pred)
print(cm)
print("Class based accuracies: ")
for i in range(3):
    print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
    
   
print("\n")
my_clf.print_tree()
