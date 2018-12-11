# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:49:19 2018

@author: liudi
"""

def index_class_num(videoname):
    with open('E:/tensorflow_workspace/two_stream/two_stream_zhuc/ucfTrainTestlist/classInd.txt','r')as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')
            split_results = line.split(' ')
            
            print(split_results)
            if split_results[1] == videoname:
                return split_results[0]
a = index_class_num('Archery')
a = int(a)
print(a)
            