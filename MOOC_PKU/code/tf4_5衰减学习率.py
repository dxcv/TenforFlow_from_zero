#coding:utf=8

import tensorflow as tf
import numpy as np

LEARNING_RATE_BASE        = 0.1
LEARNING_RATE_DECAY_STEPS = 1
LEARNING_RATE_DECAY_RATE  = 0.99

global_step = tf.Variable(0,trainable=False)

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_DECAY_STEPS,LEARNING_RATE_DECAY_RATE,staircase=True)


w =tf.Variable(tf.constant(5,dtype=tf.float32))
loss = tf.square(w+1)

#训练模型
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

with tf.Session() as sess:
    #初始化
    init=tf.global_variables_initializer()
    sess.run(init)

    #训练
    for i in range(40):
        sess.run(train_step)
        #插入测试
        print("after %5d step global_step=%lf loss=%lf  w1=%lf"
                %(i,sess.run(global_step),sess.run(loss),sess.run(w))
            )
    #训练结果
    print("\n loss = %lf  w1=%lf"%(
                sess.run(loss),sess.run(w)
            ))