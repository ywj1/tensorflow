import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data  

mnist = input_data.read_data_sets('MNIST/', one_hot=True)

in_units=784
h1_units=300
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))
#输入
x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)
#forward
hidden1=tf.nn.relu(tf.add(tf.matmul(x,W1),b1))
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
#输出
y_=tf.placeholder(tf.float32,[None,10])
#loss function
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)
#准确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#训练
batch_size=100
batch_num=int(mnist.train.num_examples/batch_size)
for i in range(10):
    for j in range(batch_num):
        x_batch,y_batch=mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:x_batch,y_:y_batch,keep_prob:0.75})
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,
                                           keep_prob:1.0}))


    
        
