import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from metrics import scores
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


#load datasets
def load(filepath):
    data =  pd.read_csv(filepath, sep=" ", dtype=float,header=None)
    data = data.drop(22, axis=1)
    data = data.drop(23, axis=1)
    
    data_np = data.to_numpy()
    
    features = data_np[:, 0:21] 
    labels = data_np[:, 21:22] 
    return features, labels


#k-fold cross validation
def training_loop(normalization,pruning,k,sampling):
    for i in range(0,k):
        #cross validation split
        X_train, X_cv , y_train, y_cv = train_test_split(train_features, train_labels, test_size=0.2, random_state=35+i)
        if normalization==True:
            print("WITH NORMALIZATION")
            #min-max normalization
            scaler = preprocessing.MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_cv = scaler.transform(X_cv)
            X_test =  scaler.transform(test_features)
        else:
            print("WITHOUT NORMALIZATION")
            X_train = X_train
            X_cv = X_cv
            X_test = test_features

        if pruning == False:
            print("WITHOUT PRUNING")
            #parameter space initializion
            criterion = ["gini", "entropy"]
            splitter = ["best","random"]
            max_depth = [10, 15, 25, 30]
            min_samples_split = [5, 10, 15]
            min_samples_leaf = [2, 10, 20] 
            max_features = ["auto", "sqrt", "log2"]
            parameter_space = dict(splitter=splitter,max_features = max_features, max_depth = max_depth,  
                min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, criterion=criterion)
            clf = DecisionTreeClassifier()
            #grid search with cross validation set
            gs = GridSearchCV(clf, parameter_space, refit=False, n_jobs=-1)
            gs.fit(X_cv, np.ravel(y_cv))
            criterion = gs.best_params_["criterion"]
            max_depth = gs.best_params_["max_depth"]
            min_samples_leaf = gs.best_params_["min_samples_leaf"]
            min_samples_split = gs.best_params_["min_samples_split"]
            max_features = gs.best_params_["max_features"]
            splitter = gs.best_params_["splitter"]
            #model fit with best parameters
            print(gs.best_params_)

            model = DecisionTreeClassifier(class_weight="balanced", random_state=35+i, criterion=criterion, 
                                           max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                           min_samples_split=min_samples_split, max_features=max_features,splitter=splitter)
            model.fit(X_train, np.ravel(y_train))
            #decision tree draw
            fig = plt.figure(figsize=(25,20))
            _ = plot_tree(model, 
                               filled=True)
            """fig.savefig(str(normalization) + str(pruning)+ str(sampling) + "decision_tree.pdf")"""
            #training set prediction
            print("PERFORMANCE ON TRAINING SET")
            y_pred = model.predict(X_train)
            cm = scores(y_train,y_pred)
            print(cm)
            print("Class based accuracies: ")
            for i in range(3):
                print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
            
            #test set prediction
            print("PERFORMANCE ON TEST SET")
        
            y_pred = model.predict(X_test)
            cm = scores(test_labels,y_pred)
            print(cm)
            print("Class based accuracies: ")
            for i in range(3):
                print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
        else:
            print("WITH PRUNING")
            
            """    
            path = model.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
        
            plt.figure(figsize=(10, 6))
            plt.plot(ccp_alphas, impurities)
            plt.xlabel("effective alpha")
            plt.ylabel("total impurity of leaves")
            clfs = []
            
            for ccp_alpha in ccp_alphas:
                clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                clf.fit(X_train, y_train)
                clfs.append(clf)
            tree_depths = [clf.tree_.max_depth for clf in clfs]
            plt.figure(figsize=(10,  6))
            plt.plot(ccp_alphas[:-1], tree_depths[:-1])
            plt.xlabel("effective alpha")
            plt.ylabel("total depth")
            from sklearn.metrics import accuracy_score
        
            acc_scores = [accuracy_score(y_cv, clf.predict(X_cv)) for clf in clfs]
            
            tree_depths = [clf.tree_.max_depth for clf in clfs]
            plt.figure(figsize=(10,  6))
            plt.grid()
            plt.plot(ccp_alphas[:-1], acc_scores[:-1])
            plt.xlabel("effective alpha")
            plt.ylabel("Accuracy scores")
            """
            #parameter space initializion
            criterion = ["gini", "entropy"]
            splitter = ["best","random"]
            max_depth = [10, 15, 25, 30]
            min_samples_split = [5, 10, 15]
            min_samples_leaf = [2, 10, 20] 
            max_features = ["auto", "sqrt", "log2"]
            ccp_alpha= [0.1, 0.005, 0.001, 0.0005]
            parameter_space = dict(splitter=splitter,max_features = max_features, max_depth = max_depth,  
                min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, criterion=criterion,ccp_alpha=ccp_alpha)
            clf = DecisionTreeClassifier()
            #grid search with cross validation set
            gs = GridSearchCV(clf, parameter_space, refit=False, n_jobs=-1)
            gs.fit(X_cv, np.ravel(y_cv))
            criterion = gs.best_params_["criterion"]
            max_depth = gs.best_params_["max_depth"]
            min_samples_leaf = gs.best_params_["min_samples_leaf"]
            min_samples_split = gs.best_params_["min_samples_split"]
            max_features = gs.best_params_["max_features"]
            splitter = gs.best_params_["splitter"]
            ccp_alpha = gs.best_params_["ccp_alpha"]
            print(gs.best_params_)
            #model fit with best parameters
            model = DecisionTreeClassifier(class_weight="balanced",random_state=35+i, criterion=criterion, 
                                           max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                           min_samples_split=min_samples_split, max_features=max_features,splitter=splitter,ccp_alpha=ccp_alpha)
            model.fit(X_train, np.ravel(y_train))
            #decision tree draw
            fig = plt.figure(figsize=(25,20))
            _ = plot_tree(model, 
                               filled=True)
            """fig.savefig(str(normalization) + str(pruning)+ str(sampling) +"decision_tree.pdf")"""
            #training set prediction
            print("PERFORMANCE ON TRAINING SET")
            y_pred = model.predict(X_train)
            cm = scores(y_train,y_pred)
            print(cm)
            print("Class based accuracies: ")
            for i in range(3):
                print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
            
            #test set prediction
            print("PERFORMANCE ON TEST SET")
        
            y_pred = model.predict(X_test)
            cm = scores(test_labels,y_pred)
            print(cm)
            print("Class based accuracies: ")
            for i in range(3):
                print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
                
                
#balance training dataset by oversampling or undersampling
def prepare_dataset(sampling):
    train_features, train_labels = load("ann-train.data")
    test_features, test_labels = load("ann-test.data")
    if sampling=="oversample":
        oversample = RandomOverSampler(sampling_strategy='minority')
        train_features, train_labels = oversample.fit_resample(train_features, train_labels)
        train_features, train_labels = oversample.fit_resample(train_features, train_labels)

        return train_features, train_labels
    elif sampling == "undersample":
        undersample = RandomUnderSampler(sampling_strategy='majority')
        train_features, train_labels = undersample.fit_resample(train_features, train_labels)
        train_features, train_labels = undersample.fit_resample(train_features, train_labels)

        return train_features, train_labels
    elif sampling == None:
        return train_features, train_labels


print("TRAINING DATASET IS USED AS IS")
test_features, test_labels = load("ann-test.data")
train_features, train_labels = prepare_dataset(None)
training_loop(normalization=0,pruning=0,k=1,sampling=0)
training_loop(normalization=0,pruning=1,k=1,sampling=0)
training_loop(normalization=1,pruning=0,k=1,sampling=0)
training_loop(normalization=1,pruning=1,k=1,sampling=0)

print("TRAINING DATASET IS BALANCED BY OVERSAMPLING")
test_features, test_labels = load("ann-test.data")
train_features, train_labels = prepare_dataset("oversample")
training_loop(normalization=0,pruning=0,k=1,sampling=1)
training_loop(normalization=0,pruning=1,k=1,sampling=1)
training_loop(normalization=1,pruning=0,k=1,sampling=1)
training_loop(normalization=1,pruning=1,k=1,sampling=1)


print("TRAINING DATASET IS BALANCED BY UNDERSAMPLING")
test_features, test_labels = load("ann-test.data")
train_features, train_labels = prepare_dataset("undersample")
training_loop(normalization=0,pruning=0,k=1,sampling=2)
training_loop(normalization=0,pruning=1,k=1,sampling=2)
training_loop(normalization=1,pruning=0,k=1,sampling=2)
training_loop(normalization=1,pruning=1,k=1,sampling=2)

