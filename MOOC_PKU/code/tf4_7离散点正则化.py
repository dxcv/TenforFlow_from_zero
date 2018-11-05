#coding:utf=8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
SEED = 233

rdm = np.random.RandomState(SEED)

Xr = rdm.randn(300,2)
Yr = [int(x0*x0 + x1*x1 < 2) for (x0,x1) in Xr]
Yrc = [['red' if Yri else 'blue'] for Yri in Yr]

Xr = np.vstack(Xr).reshape(-1,2)#整理为N行2列
Yr = np.vstack(Yr).reshape(-1,1)#整理为N行2列

# print(Xr)
# print(Yr)
# print(Yrc)

plt.scatter(Xr[:,0],Xr[:,1],c=np.squeeze(Yrc))
plt.show()



#定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b

Xnn = tf.placeholder(tf.float32,shape=(None,2))
Ynn = tf.placeholder(tf.float32,shape=(None,1))

w1 = get_weight([2,11],0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(Xnn,w1)+b1)

w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y2 = tf.nn.relu(tf.matmul(y1,w2)+b2)
Yo = y2

loss_mse = tf.reduce_mean(tf.square(Ynn-Yo))
loss_total =  loss_mse + tf.add_n(tf.get_collection('losses'))

def train1():
    #当前训练不包含正则化
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            start=(i*BATCH_SIZE)%300
            end = start + BATCH_SIZE
            sess.run(train_step,feed_dict={Xnn:Xr[start:end],Ynn:Yr[start:end]})

            #插入测试
            if i % 4000 == 0 :
                loss_mse_v = sess.run(loss_mse,feed_dict={Xnn:Xr,Ynn:Yr})
                print("After %5d steps, loss is %f"%(i,loss_mse_v))


        #画出训练分界线

        xx,yy  = np.mgrid[-3:3:0.01,-3:3:0.01]
        grid   = np.c_[xx.ravel(),yy.ravel()]
        probs1 = sess.run(Yo,feed_dict={Xnn:grid})
        probs1 = probs1.reshape(xx.shape)

        plt.scatter(Xr[:,0],Xr[:,1],c=np.squeeze(Yrc))
        plt.contour(xx,yy,probs1,levels=[.5])
        plt.show()



def train2():

    #当前训练包含正则化
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            start=(i*BATCH_SIZE)%300
            end = start + BATCH_SIZE
            sess.run(train_step,feed_dict={Xnn:Xr[start:end],Ynn:Yr[start:end]})

            #插入测试
            if i % 4000 == 0 :
                loss_v = sess.run(loss_total,feed_dict={Xnn:Xr,Ynn:Yr})
                print("After %5d steps, loss is %f"%(i,loss_v))


        #画出训练分界线

        xx,yy  = np.mgrid[-3:3:0.01,-3:3:0.01]
        grid   = np.c_[xx.ravel(),yy.ravel()]
        probs1 = sess.run(Yo,feed_dict={Xnn:grid})
        probs1 = probs1.reshape(xx.shape)

        plt.scatter(Xr[:,0],Xr[:,1],c=np.squeeze(Yrc))
        plt.contour(xx,yy,probs1,levels=[.5])
        plt.show()





STEPS=40000
train1()
train2()

