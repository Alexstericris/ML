import math
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


def create_dataset(amount,variance,step=2,correlation=None):
    val=1
    y=np.array([])
    for i in range(amount):
        y=np.append(y,val+random.randrange(-variance,variance))
        if bool(correlation) and correlation=='pos':
            val +=step
        if bool(correlation) and correlation=='neg':
            val-=step
    x=np.arange(0,len(y))
    return x,y

def get_slope_and_intercept(x,y):
    m= ((mean(x)*mean(y)-mean(x*y))/
        (mean(x)**2-mean(x**2)))
    b=mean(y)-m*mean(x)
    return m,b

def squared_error(y,y_pred):
    #doesn't matter if y-y_pred or y_pred-y
    return np.sum((y-y_pred)**2)

def coefficient_of_determination(y,y_pred):
    y_mean=np.array([mean(y)]*len(y))
    sq_sum_residuals=squared_error(y,y_pred)
    sq_sum_total=squared_error(y,y_mean)
    return 1- (sq_sum_residuals/sq_sum_total)



if __name__ == '__main__':
    x=np.array([1,2,3,4,5,6])
    y=np.array([4,6,5,7,6,8])
    m,b=get_slope_and_intercept(x,y)
    coeff=coefficient_of_determination(y,x*m+b)
    x2,y2=create_dataset(20,100,2,'pos')
    m2, b2 = get_slope_and_intercept(x2, y2)
    coeff2 = coefficient_of_determination(y2, x2 * m2 + b2)
    print(coeff2)
    plt.scatter(x,y)
    plt.scatter(x2,y2,c='red')
    # plt.xlim((0,7))
    # plt.ylim((0,10))
    plt.plot(x,x*m+b)
    plt.plot(x2,x2*m2+b2)
    plt.show()
