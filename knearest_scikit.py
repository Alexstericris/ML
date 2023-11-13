from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets
    X=X.fillna(-99999)
    print(breast_cancer_wisconsin_original.metadata)
    print(breast_cancer_wisconsin_original.variables)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf=KNeighborsClassifier()
    clf.fit(X_train,y_train.values.ravel())
    predicted=clf.predict(X_test)
    print(clf.score(X_test,y_test))

