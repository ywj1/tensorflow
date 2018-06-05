import tensorflow as tf
import time
import numpy as np
import cifar10
import cifar10_input
##定义迭代参数
max_steps = 3000 #最大迭代次数
batch_size = 128  
data_dir = '/temp/cifar10_data/cifar-10-batches-bin'#默认下载路径/home/zcm/tensorf/test/cifar10/cifar10_data/cifar-10-batches-bin/cifar-10-batches-py
##定义初始化权值函数
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))#截断到正态分布来初始化权重
    if w1 is not None:
    #w1控制L2正则化的大小
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        #L2正则化权值后再和w1相乘，用w1控制L2loss
        tf.add_to_collection('losses',weight_loss)
        #储存weight_loss到名为'loses'的collection上面
    return var
##使用cifar10下载数据集并解压展开到默认位置
cifar10.maybe_download_and_extract()#下载数据集
#训练集
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
#cifar10_input类中带的distorted_inputs()函数可以产生训练需要的数据，包括特征和label，返回封装好的tensor，每次执行都会生成一个batch_size大小的数据。
#测试集
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir=data_dir, batch_size=batch_size)
##载入数据
image_in = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])#裁剪后尺寸为24×24，彩色图像通道数为3
label_in = tf.placeholder(tf.int32, [batch_size])
##第一个卷积层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64],stddev=5e-2,w1=0.0)#5×5的卷积和，3个通道，64个滤波器
kernel1 = tf.nn.conv2d(image_in, weight1, strides=[1, 1, 1, 1], padding = 'SAME')#卷积1
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')#same?尺寸？
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
##第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64],stddev=5e-2,w1=0.0)#5×5的卷积和，第一个卷积层输出64个通道，64个滤波器
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding = 'SAME')#卷积1
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))#此处bias初始化为0.1
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')#same?尺寸？
print (pool2.shape)
##全连接层1
reshape = tf.reshape(pool2, [batch_size, -1])#将数据变为1D数据
dim = reshape.get_shape()[1].value#获取维度
print (dim)
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))#此处bias初始化为0.1
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)
##全连接层2
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))#此处bias初始化为0.1
local4 = tf.nn.relu(tf.matmul(local3,  weight4) + bias4)
##最后一层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/199.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
##计算softmax和loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    #softmax和cross entropy loss的计算合在一起
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #计算cross entropy 均值
    tf.add_to_collection('losses', cross_entropy_mean)
    #将整体losses的collection中的全部loss求和，得到最终的loss，其中包括cross entropy loss，还有后两个全连接层中weight的L2 loss
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
##数据准备
loss = loss(logits, label_in)  #传递误差和label
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) #优化器
top_k_op = tf.nn.in_top_k(logits, label_in, 1) #得分最高的那一类的准确率
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#初始化变量
tf.train.start_queue_runners()
#启动线程，在图像数据增强队列例使用了16个线程进行加速。
##训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    free, loss_value = sess.run([train_op, loss], feed_dict = {image_in: image_batch, label_in: label_batch})
    duration = time.time() - start_time #运行时间
    if step %10 == 0:
        example_per_sec = batch_size/duration#每秒训练样本数
        sec_per_batch = float(duration) #每个batch时间
        format_str = ('step %d, loss=%.2f(%.1f exaples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, example_per_sec, sec_per_batch))
##测试模型准确率
num_examples = 1000
import math
num_iter = int(math.ceil(num_examples / batch_size))#math.ceil()为向上取整
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_in: image_batch, label_in: label_batch})
##打印准确率
    true_count += np.sum(predictions)
    step +=1
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
