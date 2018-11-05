#coding:utf=8

import tensorflow as tf
import numpy as np

w =tf.Variable(tf.constant(5,dtype=tf.float32))
loss = tf.square(w+1)

#训练模型
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    #初始化
    init=tf.global_variables_initializer()
    sess.run(init)

    #训练
    for i in range(40):
        sess.run(train_step)
        #插入测试
        print("after %5d step  loss = %lf  w1=%lf"%(
                i,sess.run(loss),sess.run(w)
            ))
    #训练结果
    print("\n loss = %lf  w1=%lf"%(
                sess.run(loss),sess.run(w)
            ))