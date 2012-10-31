import simplenn
from math import *
import random

def demo_sin():
    X = []
    tmp= 0
    for i in xrange(200):
        X.append(tmp)
        tmp+=2*pi/199
    old_x = X[:]
    Y = [sin(x)+random.random()*0.1 for x in X]
    old_y = Y[:]

    X = [{'I':x} for x in X]
    Y = [{'O':y} for y in Y]

    nn = simplenn.RBFNN(['I'],['O'],nCenters=157)

    err = nn.train(X,Y,max_iter=1000)

    tmp = []

   

    for x in X:
        tmp.append(nn.test_onece(x)['O'])
   
    from matplotlib import pyplot as plt
    plt.plot(old_x,old_y,'bo')
    plt.plot(old_x,tmp,'r-')
    plt.show()

if __name__ == '__main__':
    demo_sin()