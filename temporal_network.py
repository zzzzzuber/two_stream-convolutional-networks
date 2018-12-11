import tensorflow as tf
#import pickle
import os
import numpy as np
from stacking_flow import stack_flow
import csv
from BatchNormalization import bn_layer
import config
#import pandas as pd

root_file = config.root_file
flow_file = config.flow_file

examples = 13320
epoches = 50000
is_training = True
batch_size = 10  #video nums
n_batch = examples//batch_size



def get_input_train_data():     #get input optical flow data
    for i in range(3):
        with open("E:/tensorflow_workspace/two_stream_zhuc/ucfTrainTestlist/trainlist0"+str(i+1)+'.txt','r') as frt:
            labels = frt.readlines()
        for label in labels:
            label_path = label.split(' ')[0].split('.')[0]
            label_name = (label.split('/')[0])
            video_path = os.path.join(flow_file,label_path)
            s_flow = stack_flow(video_path)  #video_name layer
            #print("get_input_Data() run correctly!")
            yield s_flow,label_name,len(s_flow)#return a video's all the optical flow images

def get_input_test_data():     #get input optical flow data
    for i in range(3):
        with open("E:/tensorflow_workspace/two_stream_zhuc/ucfTrainTestlist/testlist0"+str(i+1)+'.txt','r') as frtest:
            labels = frtest.readlines()
        for label in labels:
            label = label.split('.')[0]
            video_path = os.path.join(root_file,label)
            s_flow = stack_flow(video_path)  #video_name layer
            #print("get_input_Data() run correctly!")
            print("video_path:%s"%video_path)
            yield s_flow,label,len(s_flow)#return a video's all the optical flow images

def splitVideoFlow():
    for flow,label,leng in get_input_train_data():
        label = one_hot_label(label)
        groups = leng//10
        for i in range(groups):
            start = i*10
            end = i*10+10
            yield flow[start:end],label

'''
def get_batch_data(batch_size):
    batch_xs = []
    batch_ys = []
    #for i in range(batch_size):
        #print("第%s个batch:"%str(i))
    for flow, label, len in get_input_data():
        # print("successfully get batch data!")
        label = one_hot_label(label)
        label = copy_label(len, label)
        batch_xs.append(flow)
        batch_ys.append(label)

    print("get_batch_data() run correctly!")
    return batch_xs,batch_ys
'''

def str2vector(str):      #convert string to vector ---->one-hot label
    vector = []
    for i in str:
        if i=='0' or i=='1':
            a = int(i)
            vector.append(a)
    return vector
    
def one_hot_label(label):
    with open('E:/tensorflow_workspace/two_stream_zhuc/ucfTrainTestlist/classOnehot.csv','r') as fw:
        reader = csv.DictReader(fw)
        for row in reader:
            if row['className']==label:
                a = row['onehotLabel']
                v = str2vector(a)
    return v

def copy_label(num,label):   #deal with one label reflects number of videos problems
    labels = []
    for i in range(num):
        labels.append(label)
    l = len(label)
    labels = np.reshape(labels,[num,l])
    return labels

def xavier_weight(fan_in,fan_out,shape,constant=1): #initialize weights
    low = -constant*np.sqrt(6/(fan_in+fan_out))
    high = constant*np.sqrt(6/(fan_in+fan_out))
    initial_value = np.random.uniform(low=low,high=high,size=shape)
    return initial_value

def random_weight(shape):
    value = tf.truncated_normal(shape=shape,mean=0.0,stddev=0.001)
    return value

def fc_weights(shape):        #initialize fully connected layer weights
    initial_value = tf.truncated_normal(shape=shape,mean=0.0,stddev=0.01)
    return initial_value

def init_bias(shape):         #initialize bias
    initial_value = tf.Variable(tf.zeros(shape=shape))
    return initial_value
'''
def batch_norm(x):          #???????????????
    axis = list(range(len(x.get_shape()) - 1))
    mean,variance = tf.nn.moments(x,axis,name=None,keep_dims=True)
'''
def lrn(x,flag=is_training): #local response normalization
    if flag == True:
        value = tf.nn.lrn(input=x,depth_radius=2,bias=0.0001,alpha=1e-4,beta=1)#beta=0.75
    else:
        value = x
    return value
# 用于矩阵的全部显示
#np.set_printoptions(threshold=np.nan)

#model:temporal networks
X = tf.placeholder(tf.float32,[None,224,224,3])
Y = tf.placeholder(tf.float32,[None,101])

#5 conv layers and 2 fc and 1 softmax

#W1 = xavier_weight(3,96,[7,7,3,96])
W1 = random_weight([7,7,3,96])
b1 = init_bias([96])
conv1 = tf.nn.conv2d(X,W1,strides=[1,2,2,1],padding='SAME')+b1
conv1_pool = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
conv1_norm = bn_layer(conv1_pool,is_training)
conv1_acti = tf.nn.sigmoid(conv1_norm)


#W2 = xavier_weight(96,256,[5,5,96,256])
W2 = random_weight([5,5,96,256])
b2 = init_bias([256])
conv2 = tf.nn.conv2d(conv1_pool,W2,strides=[1,2,2,1],padding='SAME')+b2
conv2_pool = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
conv2_norm = bn_layer(conv2_pool,is_training)
conv2_acti = tf.nn.sigmoid(conv2_norm)

#W3 = xavier_weight(256,512,[3,3,256,512])
W3 = random_weight([3,3,256,512])
b3 = init_bias([512])
conv3 = tf.nn.conv2d(conv2_acti,W3,strides=[1,1,1,1],padding='SAME')+b3
conv3_norm = bn_layer(conv3,is_training)
conv3_acti = tf.nn.sigmoid(conv3_norm)

#W4 = xavier_weight(512,512,[3,3,512,512])
W4 = random_weight([3,3,512,512])
b4 = init_bias([512])
conv4 = tf.nn.conv2d(conv3_acti,W4,strides=[1,1,1,1],padding='SAME')+b4
conv4_norm = bn_layer(conv4,is_training)
conv4_acti = tf.nn.sigmoid(conv4_norm)

#W5 = xavier_weight(512,512,[3,3,512,512])
W5 = random_weight([3,3,512,512])
b5 = init_bias([512])
conv5 = tf.nn.conv2d(conv4_acti,W5,strides=[1,1,1,1],padding='SAME')+b5
conv5_pool = tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
conv5_acti = tf.nn.relu(conv5_pool)
#print(conv5_acti)
#[?,7,7,512]
conv5_reshape = tf.reshape(conv5_acti,shape=[-1,7*7*512])

W_fc1 = fc_weights([7*7*512,4096])
b_fc1 = init_bias([4096])
fc1 = tf.matmul(conv5_reshape,W_fc1)+b_fc1
fc1_norm = bn_layer(fc1,is_training)
fc1_acti = tf.nn.sigmoid(fc1_norm)

fc1_dropout = tf.nn.dropout(fc1,keep_prob=0.8)

W_fc2 = fc_weights([4096,2048])
b_fc2 = init_bias([2048])
fc2 = tf.nn.dropout(tf.matmul(fc1_dropout,W_fc2)+b_fc2,keep_prob=0.8)
fc2_norm = bn_layer(fc2,is_training)
f2_acti = tf.nn.sigmoid(fc2_norm)

W_fc3 = fc_weights([2048,101])
b_fc2 = init_bias([101])
f3 = tf.matmul(f2_acti,W_fc3)+b_fc2
y_pred = tf.nn.softmax(f3)
#print(y_pred)
'''
def loss(logits,labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')
'''

#training step 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f3,labels=Y)) #labels and logits are all shaped in [10,101]
#loss = tf.placeholder(tf.float32,[batch_size])
#momentum=0.9
#losss = loss(y_pred,Y)
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#the learning rate would be changed to 1e-3 after 50K iterations
#then 1e-4 after 70K iterations
#stop after 80K iterations
init = tf.global_variables_initializer()


with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
    saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
    sess.run(init)
    for epoch in range(1):
        print('epoch %s will start running!' % str(epoch))
        for i in splitVideoFlow():  #get 10 frames video optical flow data
            flow,label = i
            batch_xs = flow
            #label = one_hot_label(label)
            batch_ys = copy_label(10,label)
            loss = sess.run(cross_entropy,feed_dict={X:batch_xs,Y:batch_ys})
            train = sess.run(train_step,feed_dict={X:batch_xs,Y:batch_ys})
            acc = sess.run(accuracy,feed_dict={X:batch_xs,Y:batch_ys})
            print("loss:%s, acc:%s" % (str(loss),str(acc)))
            #losses.append(loss)
            #losses = tf.reduce_all(loss)
            #train = sess.run(train_step,feed_dict={loss:losses})  #每120个视频就训练一次，更新一次参数
    for j in get_input_test_data():
        test_xs,test_ys,len_label = j
        label = one_hot_label(test_ys)
        test_ys = copy_label(len_label, label)
        acc = sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys})
        print("accuracy:%s"%str(acc))
    writer.close()
    saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
    print("Model saved in file:", saver_path)



























