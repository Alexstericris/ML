import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from keras.datasets import mnist
from matplotlib import pyplot
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
import os.path

from QtUi.MainWindow import MainWindow
from support_vector_classifier import SupportVectorClassifier

if __name__ == '__main__':
    data=load_digits()
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    n_samples = len(train_X)
    reshaped=np.reshape(train_X,(n_samples,-1))
    if os.path.isfile('models/svcmodel'):
        svc=SupportVectorClassifier.load_model('models/svcmodel')
    else:
        svc=SupportVectorClassifier()
        svc.train(reshaped,train_y)
        svc.save_model('models/svcmodel')
    topredict=np.reshape(test_X,(len(test_X),-1))
    predicted=svc.predict(topredict)
    t=1
#subplotting
    # fig,axs =pyplot.subplots(nrows=1,ncols=9)
    # for i in range(9):
    #     axs[i].imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    # pyplot.show()

