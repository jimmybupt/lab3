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

for i = 1 to cross_fold
	size = block_size
	if i == cross_fold:
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
	data = ast.literal_eval(line);
	for D in data:
		S[i, int(D[0])] = float(D[1])
	i=i+1
	if i==rdim:
		break
label = []
i = 0
for line in label_file:
	label.append(ast.literal_eval(line))
	if i==rdim:
		break

print "done"
S = S.tocsr()

data_process_time= time.time()  - start_time

#identify training set / test set



#train model

#test model

#evaluate model
