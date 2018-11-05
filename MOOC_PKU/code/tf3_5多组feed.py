#coding:utf=8
#两层简单神经网络(全连接)
import tensorflow as tf

#定义输入和参数

x=tf.placeholder(tf.float32,shape=(None,2))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,mean=0,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,mean=0,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#会话
with tf.Session() as sess:
    #初始化
    init=tf.global_variables_initializer()
    #运行
    sess.run(init)
    o=sess.run(y,feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]})
    print("o:\n",o)
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))
