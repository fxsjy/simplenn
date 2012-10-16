import simplenn
from math import *
import random

def demo_xor():
    
    inputs = [
        {'input1':0,'input2':0},
        {'input1':0,'input2':1},
        {'input1':1,'input2':0},
        {'input1':1,'input2':1}
    ]

    targets = [
        {'output':0},
        {'output':1},
        {'output':1},
        {'output':0}
    ]

    nn = simplenn.RBFNN(['input1','input2'],['output'],nCenters=len(inputs))

    err = nn.train(inputs,targets,max_iter=1000)

    print 'test...'
    for t in inputs:
        print t, round(nn.test_onece(t)['output'])

if __name__ == '__main__':
    demo_xor()
