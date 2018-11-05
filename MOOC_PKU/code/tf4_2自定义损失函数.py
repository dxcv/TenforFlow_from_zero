#coding:utf=8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455
learning_rate = 0.001
momentum=0.9
SETEPS=5000
COST=1
PROFIT=9
# 生成数据集
rdm = np.random.RandomState(SEED)
Xr = rdm.rand(32,2)
Yr = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in Xr]


x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,1],stddev=1,mean=0,seed=1))



#定义前向传播过程
y=tf.matmul(x,w1)


#定义损失函数
loss = tf.reduce_mean(tf.square(y_-y))
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),COST*(y-y_),PROFIT*(y_-y)))

train_step = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#训练模型
with tf.Session() as sess:
    #初始化
    init=tf.global_variables_initializer()
    sess.run(init)

    #训练
    for i in range(SETEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:Xr[start:end],y_:Yr[start:end]})
    #插入测试
        if i % 500 == 0:
            loss_tmp = sess.run(loss,feed_dict={x:Xr,y_:Yr})
            print("after %5d step,loss = %lf"%(i,loss_tmp))
            print("w1:",sess.run(w1))
            print("\n")
    #训练结果
    print("\n")
    print("w1:\n",sess.run(w1))
