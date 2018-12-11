import tensorflow as tf
import csv
import config
import os

def get_all_class(filepath):
    labels = []
    name = []
    with open(filepath,'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            label = str(line).split(' ')[0]
            n = str(line).split(' ')[1].split('\\')[0]
            name.append(n)
            #print(name)
            #label = int.from_bytes(b'1', byteorder='little')
            num = label.split("'")[1]
            #print(num)
            #print(type(int(num)))
            labels.append(int(num))

    with tf.Session() as sess:
        batch_size = tf.size(labels) #value = 101
        for i in range(101):
            labels[i] = i
        labels = tf.expand_dims(labels,1) #vector ----> R[101x1] range:1-101

        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)#vector ----> R[101x1] range:0-100

        concated = tf.concat([indices, labels],1)
        #print(sess.run(concated))
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 101]), 1, 0)
        onehot = sess.run(onehot_labels)
        p = os.path.join(config.split_file,'classOnehot.csv')
        with open(p,'w',newline='')as fw:

            writer = csv.writer(fw,dialect=("excel"))

            a = "classInd".encode(encoding='utf-8')
            b = "className".encode(encoding='utf-8')
            c = "onehotLabel".encode(encoding='utf-8')
            #a = bytes("classInd",encoding='utf-8')
            writer.writerow([a,b,c])
            #writer.writerow([a])

            for i in range(101):
                x = name[i]
                y = list(onehot[i])
                writer.writerow([i,x,y])

            print("successfully!")

path = os.path.join(config.split_file,"classInd.txt")
get_all_class(path)
