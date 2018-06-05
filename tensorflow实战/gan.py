import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

#Hyper Parameters

batch_size=64
lr_g=0.0001 #learning rate for generator
lr_d=0.0001 #learning rate for discriminator
n_ideas = 5
art_components=15
paint_points = np.vstack([np.linspace(-1,1,art_components) for _ in range(batch_size)])

plt.plot(paint_points[0], 2 * np.power(paint_points[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(paint_points[0], 1 * np.power(paint_points[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()

def artist_works():
    a = np.random.uniform(1,2,size=batch_size)[:,np.newaxis]
    paintings=a*np.power(paint_points,2)+a-1
    return paintings

with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32,[None,n_ideas])
    G_l1=tf.layers.dense(G_in,128,tf.nn.relu)
    G_out = tf.layers.dense(G_l1,art_components)

with tf.variable_scope('Discriminator'):
    real_art=tf.placeholder(tf.float32,[None,art_components],name='real_in')
    D_l0=tf.layers.dense(real_art,128,tf.nn.relu,name='l')
    prob_artist0 = tf.layers.dense(D_l0,1,tf.nn.sigmoid,name='out')
    D_l1=tf.layers.dense(G_out,128,tf.nn.relu,name='l',reuse=True)
    prob_artist1=tf.layers.dense(D_l1,1,tf.nn.sigmoid,name='out',reuse=True)

D_loss = -tf.reduce_mean(tf.log(prob_artist0)+tf.log(1-prob_artist1))
G_loss=tf.reduce_mean(tf.log(1-prob_artist1))

train_D=tf.train.AdamOptimizer(lr_d).minimize(
    D_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))

train_G = tf.train.AdamOptimizer(lr_g).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()
for step in range(5000):
    artist_paintings=artist_works()
    G_ideas = np.random.rand(batch_size,n_ideas)
    G_paintings,pa0,D1,_,_ = sess.run([G_out,prob_artist0,D_loss,train_D,train_G],
                                      feed_dict={G_in:G_ideas,real_art:artist_paintings})

    if step%50==0:
        plt.cla();
        plt.plot(paint_points[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(paint_points[0], 2 * np.power(paint_points[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(paint_points[0], 1 * np.power(paint_points[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D1, fontdict={'size': 15})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=12);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()
