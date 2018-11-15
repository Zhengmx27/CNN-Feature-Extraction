import numpy as np
import tensorflow as tf
import random


def loadTrain_x(filePath):
    f = open(filePath,'r')
    for line in f.readlines():
        lineArr = line.strip().split()
        train_x.append(lineArr[0:])
    f.close()
    for i in range(len(train_x)):
        for j in range(len(train_x[i])):
            train_x[i][j] = float(train_x[i][j])


def loadTrain_y(filePath):
    f = open(filePath,'r')
    for line in f.readlines():
        lineArr = line.strip().split()
        train_y.append(lineArr[1])
    f.close()
    for i in range(len(train_y)):
        train_y[i] = int(train_y[i])


def loadTest_x(filePath):
    f = open(filePath,'r')
    for line in f.readlines():
        lineArr = line.strip().split()
        test_x.append(lineArr[0:])
    f.close()
    for i in range(len(test_x)):
        for j in range(len(test_x[i])):
            test_x[i][j] = float(test_x[i][j])

def loadTest_y(filePath):
    f = open(filePath,'r')
    for line in f.readlines():
        lineArr = line.strip().split()
        test_y.append(lineArr[0:])
    f.close()
    for i in range(len(test_y)):
        for j in range(len(test_y[i])):
            test_y[i][j] = int(test_y[i][j])




#加载数据

train_x = []
train_y = []

test_x = []
test_y = []


loadTrain_x('cnn90_N2V_W100.txt')
loadTrain_y('Cora_category.txt')
loadTest_x('else_data_80%.txt')
loadTest_y('else_lable_80%_vec.txt')


y = []
for i in range(len(train_y)):
    temp = [0 for j in range(7)]
    temp[train_y[i]] = 1
    y.append(temp)


train_x = np.array(train_x)
train_y = np.array(y)
test_x = np.array(test_x)
test_y = np.array(test_y)


x = tf.placeholder(tf.float32,[None, 256])
y_ = tf.placeholder(tf.float32, [None, 7])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#***********************在这里进行reshape 变成符合卷积神经网络的输入
x_image = tf.reshape(x, [-1,16,16,1])


W_conv1 = weight_variable([5,5,1,32])


b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)





W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)




W_fc1 = weight_variable([4*4*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)



#-dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024,7])
b_fc2 = bias_variable([7])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(1000):
       
        batch = ([], [])
        p = random.sample(range(2708), 100)
        for k in p:
            batch[0].append(train_x[k])
            batch[1].append(train_y[k])
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, train accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})
    test_accuracy = accuracy.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.})
    print('the test accuracy :{}'.format(test_accuracy))

    feature = sess.run(h_fc1, feed_dict={x: test_x})
    print(type(feature))
    print(feature.shape)



