import simplenn
import random

def argmax(D):
	tmp = max([(v,k) for k,v in D.iteritems()])
	return tmp[1]

content = open("iris_proc.data").read()
lines=content.split("\n")
r_inputs = []
r_outputs = []

def processX(raw_x):
	return dict([(i,float(v)) for i,v in enumerate(raw_x)])

def processY(raw_y):
	d={0:0,1:0,2:0}
	d[int(raw_y)]=1
	return d

for line in lines:
	line = line.strip()
	if line=='': break
	tup = line.split(",")
	r_inputs.append(processX(tup[:-1]))
	r_outputs.append(processY(tup[-1]))

idx_list = range(len(r_inputs))
random.shuffle(idx_list)
p = len(idx_list) - len(idx_list)/3

inputs = [ r_inputs[i] for i in idx_list[:p] ]
outputs = [r_outputs[i] for i in idx_list[:p] ]

test_inputs = [ r_inputs[i] for i in idx_list[p:] ]
test_outputs = [r_outputs[i] for i in idx_list[p:] ]

#print inputs,outputs
#print test_inputs,test_outputs
nn = simplenn.RBFNN(range(4),range(3),nCenters=len(inputs))

nn.train(inputs,outputs,sm=True)

wrong = 0
for i,t in enumerate(test_inputs):
	t_out = argmax(nn.test_onece(t))
	if t_out != argmax(test_outputs[i]):
		print t_out,test_outputs[i]
		wrong+=1

print 'correct rate %.2f' % (1-float(wrong)/len(test_outputs))




