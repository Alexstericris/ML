import numpy as np
import pickle

class SupportVectorClassifier:
    classifiers = []
    learning_rate=0.01
    lambda_param=0.01
    num_iterations=100
    def __init__(self, classifiers=None, learning_rate=0.01, lambda_param=0.01, num_iterations=100):
        if classifiers is None:
            classifiers = []
        self.classifiers=classifiers
        self.learning_rate=learning_rate
        self.lambda_param=lambda_param
        self.num_iterations=num_iterations
    def train(self,X,y):
        X=np.array(X)
        y=np.array(y)
        num_samples,num_features=X.shape
        unique_classes=np.unique(y)
        for label in unique_classes:
            binary_labels=np.where(y==label,1,-1)
            w=np.zeros(num_features)
            b=0
            for i in range(self.num_iterations):
                if i==10:
                    t=1
                scores=np.dot(X,w)-b
                margins=binary_labels*scores
                misclassified=np.where(margins<1)[0]
                w-=self.learning_rate*\
                   (w*self.lambda_param*w-np.dot(X[misclassified].T,binary_labels[misclassified]))\
                   /num_samples
                b-=self.learning_rate*np.mean(-binary_labels[misclassified])

            self.classifiers.append((w, b))

    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.classifiers)
        scores = np.zeros((num_samples, num_classes))

        # Calculate scores for each binary classifier
        for i, (w, b) in enumerate(self.classifiers):
            scores[:, i] = np.dot(X, w) - b

        # Predict the class with the highest score
        return np.argmax(scores, axis=1)

    def save_model(self, filename):
        model_params = {
            'classifiers': self.classifiers,
            'learning_rate': self.learning_rate,
            'lambda_param': self.lambda_param,
            'num_iterations': self.num_iterations,
        }
        with open(filename, 'wb') as file:
            pickle.dump(model_params, file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            model_params = pickle.load(file)
        model = SupportVectorClassifier(
            model_params['classifiers'],
            model_params['learning_rate'],
            model_params['lambda_param'],
            model_params['num_iterations'],
        )
        return model