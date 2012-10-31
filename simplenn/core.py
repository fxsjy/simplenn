import random
import math
#import pprint

def sigmoid(x):
	return 1.0/(1 + math.e**(-x))


def dsigmoid(y):
	return y*(1.0-y)


class NN():
	def __init__(self,inputs_keys,outputs_keys):
		self.weight_matix = {}
		self.lastchange = {}
		self.inputs_keys = inputs_keys
		self.inputs_keys.append("~") #bias node
		self.outputs_keys = outputs_keys
		for ip in inputs_keys:
			self.weight_matix[ip] = dict([(op,-2.0+random.random()*4.0) for op in outputs_keys])
			self.lastchange[ip] = dict([(op,0) for op in outputs_keys])

	def train_onece(self, X,Y,sm,eta=0.25,beta=0.1):
		X['~']=1.0
		if sm :
			Y_ = dict([ (k, sigmoid(sum([ X[i]* self.weight_matix[i][k] for i in X.keys() ])) ) for k in Y.keys() ] )
			error = dict([(k,Y_[k]-v) for k,v in Y.iteritems()])
			for k2 in Y:
				dsm = dsigmoid(Y_[k2])
				for k1 in X:
					delta = dsm* error[k2]*X[k1]
					change = delta*eta+ self.lastchange[k1][k2]*beta
					self.weight_matix[k1][k2] -= change
					self.lastchange[k1][k2] = change
		else:
			Y_ = dict([ (k, sum([ X[i]* self.weight_matix[i][k] for i in X.keys() ]) ) for k in Y.keys() ] )
			error = dict([(k,Y_[k]-v) for k,v in Y.iteritems()])
			for k1 in X:
				for k2 in Y:
					delta = error[k2]*X[k1]
					change = delta*eta + self.lastchange[k1][k2]*beta
					self.weight_matix[k1][k2] -= change
					self.lastchange[k1][k2] = change
		return 0.5*sum([i**2 for i in error.values()])

	def train(self,inputs,outputs,sm=False):
		err = 0.0
		if sm==False:
			eta,beta = 0.025,0.01
		else:
			eta,beta = 0.25,0.1
		for i, x in enumerate(inputs):
			y = outputs[i]
			err += self.train_onece(x,y,sm,eta=eta,beta=beta)
		#pprint.pprint(self.weight_matix)
		return err

	def test_onece(self, X,sm=False):
		X['~']=1.0
		if sm==False:
			Y_ = dict([ (k, sum([ X[i]* self.weight_matix[i][k] for i in X.keys() ]) ) for k in self.outputs_keys ] )
		else:
			Y_ = dict([ (k, sigmoid(sum([ X[i]* self.weight_matix[i][k] for i in X.keys() ])) ) for k in self.outputs_keys ] )
		return Y_

class RBFNN():
	def __init__(self,inputs_keys,outputs_keys,nCenters):
		self.outputs_keys = outputs_keys
		idx_list = range(nCenters)
		random.shuffle(idx_list)
		self.center_idx_list = idx_list

	def make_rbf(self,p,c):
		k1_set = p.keys()
		k2_set = c.keys()
		union_set = set(k1_set) | set(k2_set)
		dist = sum([ ( p.get(k,0.0)-c.get(k,0.0) )**2 for k in union_set])
		return math.exp(-8*dist)

	def train(self,inputs,outputs,max_iter=1000,sm=False):
		new_inputs = []
		self.centers = [inputs[i] for i in self.center_idx_list]
		for i,p in enumerate(inputs):
			new_inputs.append({})
			div = 0.0
			for j,c in enumerate(self.centers):
				new_inputs[i][j] = self.make_rbf(p,c)
				div += new_inputs[i][j] 
			for j,c in enumerate(self.centers):
				new_inputs[i][j] /= div
		#print new_inputs
		self.nn = NN(range(len(self.centers)),self.outputs_keys)
		#pprint.pprint(new_inputs,open("ip.txt","wb"))
		#.pprint(outputs,open("op.txt","wb"))
		change = range(len(new_inputs))
		old_err1 = 1.0
		old_err2 = 2.0
		for j in xrange(max_iter):
			new_inputs = [new_inputs[i] for i in change]
			outputs = [outputs[i] for i in change]
			err = self.nn.train(new_inputs,outputs,sm=sm)
			if j%100==0:
				print 'loop:',j,'err',err
				if (old_err2-old_err1<0.001) and (old_err1-err<0.001):
					break #early stopping
			random.shuffle(change)
			old_err2 = old_err1
			old_err1 = err

	def test_onece(self, X,sm=False):
		d={}
		div = 0.0
		for j,c in enumerate(self.centers):
			d[j] = self.make_rbf(X,c)
			div += d[j]
		for j,c in enumerate(self.centers):
			d[j] /= div
		Y_ = self.nn.test_onece(d,sm)
		#print Y_
		return Y_
