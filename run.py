print 'CSE 5243 Prediction Analysis by Kun Liu & Zhe Dong'

#program options
#from optparse import OptionParser
#import options
#parser = OptionParser()
#options.initialize_parser(parser)
#(options, args) = parser.parse_args()

#load data file

vector_file = open('vector1.txt', 'r')
label_file = open('label.txt', 'r')

import ast
import sys
import time
from scipy.sparse import *
info = open("info.txt",'r');
rdim = int(info.readline())
cdim = int(info.readline())

cross_fold = 5 #number of blocks for cross validation
block_size = rdim / cross_fold
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

initial_time = time.time()

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

#data_process_time= time.time()  - start_time

#identify training set / test set
import classifier
import numpy
from sklearn.metrics import precision_recall_fscore_support

offline_timelist_gnb=[]
online_timelist_gnb=[]
offline_timelist_clf=[]
online_timelist_clf=[]
accuracy_gnb=[]
accuracy_clf=[]

for i in range (0, cross_fold):
	Idx = range(0, cross_fold)
	Idx.pop(i)
	tmp = []
	Training_Label = []
	for j in Idx:
		tmp.append(T[j])
		Training_Label += L[j]
	Training_Data = vstack(tmp)
	Test_Data = T[i]
	Test_Label = L[i]

    #naive bayes classification
  	print ("") 
  	print ("cross validation trial: "+str(i+1)+" out of "+str(cross_fold)) 
   
	gnb, offline_time_gnb = classifier.bayes_classifier(Training_Data.toarray(), numpy.array(Training_Label, dtype=str))
	offline_timelist_gnb.append(offline_time_gnb)
  	score_gnb = gnb.score(Test_Data.toarray(), numpy.array(Test_Label, dtype=str))   
  	accuracy_gnb.append(score_gnb) 
   
	print ("  offline time cost is: " + str(offline_time_gnb))
  
 	start_time=time.time()
     	predict_gnb=gnb.predict(Test_Data.toarray())
	online_time_gnb=time.time()-start_time
	online_timelist_gnb.append(online_time_gnb)
 
	print ("  online time cost is: " + str(online_time_gnb/block_size))
 	print ("  accuracy is: " +  str(score_gnb))
  	print ("  precision, recall, fscore and support are: ")
  	print (precision_recall_fscore_support(numpy.array(Test_Label, dtype=str),predict_gnb,average='macro'))

	clf, offline_time_clf = classifier.tree_classifier(Training_Data.toarray(), numpy.array(Training_Label, dtype=str))
  	offline_timelist_clf.append(offline_time_clf)
 	score_clf = clf.score(Test_Data.toarray(), numpy.array(Test_Label, dtype=str))  
  	accuracy_clf.append(score_clf) 
   
	print ("  offline time cost is: " + str(offline_time_clf))
 
  	start_time=time.time()
     	predict_clf=clf.predict(Test_Data.toarray())
	online_time_clf=time.time()-start_time
	online_timelist_clf.append(online_time_clf) 
 
	print ("  online time cost is: " + str(online_time_clf/block_size))
 	print ("  accuracy is: " +  str(score_clf))
  	print ("  precision, recall, fscore and support are: ") 
  	print (precision_recall_fscore_support(numpy.array(Test_Label, dtype=str),clf.predict(Test_Data.toarray()),average='macro'))

#print average time cost for training and testing 
print ("") 
print ("average offline cost for Naive Bayes classification is: " +str(sum(offline_timelist_gnb)/cross_fold))
print ("average offline cost for decision tree classification is: " +str(sum(offline_timelist_clf)/cross_fold))
print ("average online cost for Naive Bayes classification is: " +str(sum(online_timelist_gnb)/rdim))
print ("average online cost for decision tree classification is: " +str(sum(online_timelist_clf)/rdim))
print ("average accuracy for Naive Bayes classification is: " +str(sum(accuracy_gnb)/cross_fold))
print ("average accuracy for decision tree classification is: " +str(sum(accuracy_clf)/cross_fold))
print ("total running time is : " +str(time.time()-initial_time))
