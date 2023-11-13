import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def k_nearest(data, predict, k=3):
    if len(data) >=k:
        warnings.warn('k is set to a value less than total voting groups')
    distances=[]
    for group in data:
        for features in data[group]:
            eucl_dist=np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([eucl_dist,group])
    #closest k points
    votes=[row[1] for row in sorted(distances)[:k]]
    vote_result=Counter(votes).most_common(1)[0][0]
    return vote_result

if __name__ == '__main__':
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    X = breast_cancer_wisconsin_original.data.features
    X=X.fillna(-9999)
    y = breast_cancer_wisconsin_original.data.targets
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
    train_set={2:[],4:[]}
    test_set={2:[],4:[]}
    for i in range(X_train.shape[0]):
        train_set[y_train.iloc[i][0]].append(X_train.iloc[i])
    for i in range(X_test.shape[0]):
        test_set[y_test.iloc[i][0]].append(X_test.iloc[i])

    correct=0
    total=0
    for group in test_set:
        for predict in test_set[group]:
            vote=k_nearest(train_set,predict,5)
            if group ==vote:
                correct+=1
            total+=1

    accuracy=correct/total
    print(accuracy)
    # t=k_nearest(train_set,,3)
    X_class_2 = X[np.array(y==2)]
    X_class_4 = X[np.array(y==4)]
    plt.scatter(X_class_2['Clump_thickness'], X_class_2['Uniformity_of_cell_shape'], c="red", marker='o', edgecolor='k', s=50)
    plt.scatter(X_class_4['Clump_thickness'], X_class_4['Uniformity_of_cell_shape'], c="blue", cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Breast Cancer Wisconsin Dataset')
    plt.colorbar(label='Target Class')
    plt.show()
    # X=X.fillna(-99999)
    # print(breast_cancer_wisconsin_original.metadata)
    # print(breast_cancer_wisconsin_original.variables)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # clf=KNeighborsClassifier()
    # clf.fit(X_train,y_train.values.ravel())
    # predicted=clf.predict(X_test)
    # print(clf.score(X_test,y_test))

