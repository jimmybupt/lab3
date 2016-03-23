print 'CSE 5243 Prediction Analysis by Kun Liu & Zhe Dong'

#program options
from optparse import OptionParser
import options
parser = OptionParser()
options.initialize_parser(parser)
(options, args) = parser.parse_args()

#load data file

vector_file = open(options.in_file, 'r')
label_file = open('label.txt', 'r')

import ast
import sys
from scipy.sparse import *
info = open("info.txt",'r');
rdim = int(info.readline())
cdim = int(info.readline())

cross_fold = 5
block_size = rdim / 5
S = []	#vectors
L = []	#labels

for i in range(0, cross_fold):
	size = block_size
	if i == cross_fold - 1:
		size = rdim - ((cross_fold-1)*block_size)
	M = lil_matrix((size, cdim))
	V = [None] * size;
	S.append(M)
	L.append(V)

import time
start_time = time.time()

print "Reading vectors... ",
sys.stdout.flush()
i = 0
for line in vector_file:
	block = i / block_size
	if block >= cross_fold:
		block = cross_fold - 1
	offset = i - block_size * block

	data = ast.literal_eval(line);
	for D in data:
		S[block][offset, int(D[0])] = float(D[1])
	i=i+1
	if i==rdim:
		break

i = 0
for line in label_file:
	block = i / block_size
	if block >= cross_fold:
		block = cross_fold - 1
	offset = i - block_size * block

	label = ast.literal_eval(line)
	#print label[0].encode('ascii')
	L[block][offset] = str(label[0])
	i = i+1
	if i==rdim:
		break

T = []
for M in S:
	T.append(M.tocsr())	

print "done"

data_process_time= time.time()  - start_time

#identify training set / test set

Training_Data = vstack([T[0],T[1],T[2],T[3]])
Training_Label = L[0] + L[1] + L[2] + L[3]
Test_Data = T[4]
Test_Label = L[4]

#train model
import classifier
import numpy

#print Training_Label
#print numpy.array(Training_Label)
Classifier , build_time = classifier.KNN_classifier(Training_Data.toarray(), numpy.array(Training_Label, dtype=str))

#test model
#Predict_Label = Classifier.predict(Test_Data.toarray())

#print Predict_Label

#evaluate model
score = Classifier.score(Test_Data.toarray(), numpy.array(Test_Label, dtype=str))

print score
