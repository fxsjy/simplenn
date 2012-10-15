import simplenn

def R(n):
    return {'output':n}

def Q(L):
    H = {}
    for x in L:
        if not x in H: H[x]=0
        H[x]+=1
    return H

def demo_digit():
    patterns = [
        Q([7,1,1,1]),
        Q([8,8,0,9]),
        Q([2,1,7,2]),
        Q([6,6,6,6]),
        Q([1,1,1,1]),
        Q([2,2,2,2]),
        Q([7,6,6,2]),
        Q([9,3,1,3]),
        Q([0,0,0,0]),
        Q([5,5,5,5]),
        Q([8,1,9,3]),
        Q([8,0,9,6]),
        Q([4,3,9,8]),
        Q([9,4,7,5]),
        Q([9,0,3,8]),
        Q([3,1,4,8])
    ]

    targets = [R(0),R(6),R(0),R(4),R(0),R(0),R(2),R(1),R(4),R(0),R(3),R(5),R(3),R(1),R(4),R(2)]
    # create a network with two input, two hidden, and one output nodes
    n = simplenn.NN(range(10), ['output'])
    # train it with some patterns
    for i in xrange(1000):
        err = n.train(patterns,targets,sm=False)
        if i % 100 ==0:
            print "error", err

    patterns.append(Q([0,2,4,8]))
    patterns.append(Q([2,5,8,1]))
    for pat in patterns:
        print round(n.test_onece(pat)['output'])

    #tt = patterns
    #print list(n.test_num(patterns))
    

if __name__ == '__main__':
    demo_digit()

