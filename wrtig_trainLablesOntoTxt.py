from my_cnnModel import *

f=open('train_labels.txt','w')
f.write(repr(training_set.class_indices)+'\n')
f.close()