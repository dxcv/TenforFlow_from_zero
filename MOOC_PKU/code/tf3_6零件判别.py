#coding:utf=8
#两层简单神经网络(全连接)
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455
learning_rate = 0.001
momentum=0.9
SETEPS=3000

#基于seed产生随机数
rng = np.random.RandomState(SEED)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
Xr = rng.rand(32,2)
#从X这个32行2列的矩阵中 取出一行，判断如果小于1 给Y赋值1 反之0
#作为输入数据集的标签
Yr = [[int(x0+x1<1)] for (x0,x1) in Xr]

# print("rng:",rng)
# print("X:",X)
# print("Y:",Y)


x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,mean=0,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,mean=0,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)


#定义损失函数
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
    #训练结果
    print("\n")
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))