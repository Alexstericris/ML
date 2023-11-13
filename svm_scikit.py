from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn import svm

if __name__ == '__main__':
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
    # df=breast_cancer_wisconsin_original.data
    X = breast_cancer_wisconsin_original.data.features
    y = breast_cancer_wisconsin_original.data.targets
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    # X = X.fillna(-99999)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf=svm.SVC()
    clf.fit(X_train,y_train.values.ravel())
    predicted=clf.predict(X_test)
    print(clf.score(X_test,y_test))

