#coding:utf=8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b

def forward(x,regularizer):
    w1 = get_weight([2,11],0.01)
    b1 = get_bias([11])

    w2 = get_weight([11,1],0.01)
    b2 = get_bias([1])

    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    y2 = tf.nn.relu(tf.matmul(y1,w2)+b2)

    Yo = y2

    return Yo

def backward():
    global_step = tf.Variable(0,trainable=False)
    Xi = tf.placeholder(tf.float32,shape=(None,2))
    Yi = tf.placeholder(tf.float32,shape=(None,1))
    REGULARIZER = 0.01
    Yo = forward(Xi,REGULARIZER)


    loss_mse = tf.reduce_mean(tf.square(Yo-Yi))
    # loss_cem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y,labels=Yi))
    lossA = loss_mse
    # lossB = loss_cem
    lossC = loss_mse + tf.add_n(tf.get_collection('losses'))
    # lossD = loss_cem + tf.add_n(tf.get_collection('losses'))
    loss  = lossC


    LEARNING_RATE_BASE        = 0.1
    LEARNING_RATE_DECAY_STEPS = 1000
    LEARNING_RATE_DECAY_RATE  = 0.99
    learning_rate_exponential_decay = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_DECAY_STEPS,LEARNING_RATE_DECAY_RATE,staircase=True)

    learning_rateA = learning_rate_exponential_decay
    learning_rateB = 0.001
    learning_rate  = learning_rateB



    train_step_GradientDescentOptimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    MOMENTUM=0.9
    train_step_MomentumOptimizer = tf.train.MomentumOptimizer(learning_rate,MOMENTUM).minimize(loss,global_step=global_step)
    train_step_AdamOptimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    train_stepA = train_step_GradientDescentOptimizer
    train_stepB = train_step_MomentumOptimizer
    train_stepC = train_step_AdamOptimizer

    train_step  = train_step_AdamOptimizer



    # ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # ema_op = ema.apply(tf.trainable_variables())
    # with tf.control_dependencies([train_step,ema_op]):
    #     train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        #初始化
        init=tf.global_variables_initializer()
        sess.run(init)


        for i in range(STEPS):
            start=(i*BATCH_SIZE)%300
            end = start + BATCH_SIZE
            sess.run(train_step,feed_dict={Xi:Xr[start:end],Yi:Yr[start:end]})

            #插入测试
            if i % 2000 == 0 :
                loss_v = sess.run(loss_mse,feed_dict={Xi:Xr,Yi:Yr})
                print("After %5d steps, loss is %f"%(i,loss_v))

        #画出训练分界线

                xx,yy  = np.mgrid[-3:3:0.01,-3:3:0.01]
                grid   = np.c_[xx.ravel(),yy.ravel()]
                probs = sess.run(Yo,feed_dict={Xi:grid})
                probs = probs.reshape(xx.shape)
                plt.scatter(Xr[:,0],Xr[:,1],c=np.squeeze(Yrc))
                plt.contour(xx,yy,probs,levels=[.5])
                plt.show()



def generateds():

    rdm = np.random.RandomState(SEED)
    Xr  = rdm.randn(300,2)
    Yr  = [int(x0*x0 + x1*x1 < 2) for (x0,x1) in Xr]
    Yrc = [['red' if Yri else 'blue'] for Yri in Yr]
    Xr  = np.vstack(Xr).reshape(-1,2)#整理为N行2列
    Yr  = np.vstack(Yr).reshape(-1,1)#整理为N行1列

    return Xr,Yr,Yrc

if __name__ == '__main__':

    BATCH_SIZE = 30
    SEED       = 233
    STEPS      = 40001

    Xr,Yr,Yrc = generateds()

    plt.scatter(Xr[:,0],Xr[:,1],c=np.squeeze(Yrc))
    plt.show()

    backward()