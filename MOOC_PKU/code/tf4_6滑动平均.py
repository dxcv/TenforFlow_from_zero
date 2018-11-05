#coding:utf=8

import tensorflow as tf
import numpy as np



w1 = tf.Variable(0,dtype=tf.float32)
global_step = tf.Variable(0,trainable=False)

# 实现滑动平均类 衰减率0.99 当前轮数global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    #初始化
    init=tf.global_variables_initializer()
    sess.run(init)

    print("\n w1 = %lf  ema.w1=%lf"%(
              sess.run(w1),sess.run(ema.average(w1))
            ))

    sess.run(tf.assign(w1,1))
    sess.run(ema_op)

    print("\n w1 = %lf  ema.w1=%lf"%(
              sess.run(w1),sess.run(ema.average(w1))
            ))

    sess.run(tf.assign(global_step,100))
    sess.run(tf.assign(w1,10))
    sess.run(ema_op)

    print("\n w1 = %lf  ema.w1=%lf"%(
              sess.run(w1),sess.run(ema.average(w1))
            ))

    sess.run(ema_op)
    print("\n w1 = %lf  ema.w1=%lf"%(
          sess.run(w1),sess.run(ema.average(w1))
        ))

    sess.run(ema_op)
    print("\n w1 = %lf  ema.w1=%lf"%(
          sess.run(w1),sess.run(ema.average(w1))
        ))

    sess.run(ema_op)
    print("\n w1 = %lf  ema.w1=%lf"%(
          sess.run(w1),sess.run(ema.average(w1))
        ))

