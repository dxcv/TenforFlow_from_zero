#coding:utf=8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

#定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape,regularizer):
    print("get_weight")
    w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    print("get_bias")
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    print("forward")

    w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b1 = get_bias([LAYER1_NODE])

    w2 = get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
    b2 = get_bias([OUTPUT_NODE])


    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    y2 = tf.matmul(y1,w2)+b2

    Yo = y2

    return Yo

def backward(mnist):
    print("backward")
    global_step = tf.Variable(0,trainable=False)
    Xi = tf.placeholder(tf.float32,shape=(None,INPUT_NODE))
    Yi = tf.placeholder(tf.float32,shape=(None,OUTPUT_NODE))
    Yo = forward(Xi,REGULARIZER)


    # loss_mse = tf.reduce_mean(tf.square(Yo-Yi))
    loss_mse = tf.reduce_mean(tf.square(tf.cast(tf.equal(tf.argmax(Yo, 1), tf.argmax(Yi, 1)),tf.float32)-1.0))
    loss_cem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yo,labels=tf.argmax(Yi,1)))
    lossA = loss_mse
    lossB = loss_cem
    lossC = loss_mse + tf.add_n(tf.get_collection('losses'))
    lossD = loss_cem + tf.add_n(tf.get_collection('losses'))
    loss  = lossD



    learning_rate_exponential_decay = tf.train.exponential_decay(
                                        LEARNING_RATE_BASE,
                                        global_step,
                                        mnist.train.num_examples/BATCH_SIZE,
                                        LEARNING_RATE_DECAY_RATE,
                                        staircase=True)
    learning_rateA = learning_rate_exponential_decay
    learning_rateB = 0.01
    learning_rate  = learning_rateA



    train_step_GradientDescentOptimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    MOMENTUM=0.9
    train_step_MomentumOptimizer = tf.train.MomentumOptimizer(learning_rate,MOMENTUM).minimize(loss,global_step=global_step)
    train_step_AdamOptimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    train_stepA = train_step_GradientDescentOptimizer
    train_stepB = train_step_MomentumOptimizer
    train_stepC = train_step_AdamOptimizer

    train_step  = train_stepA


    #滑动平均

    if USE_EMA__ :
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step,ema_op]):
            train_op = tf.no_op(name='train')
    else:
        train_op = train_step




    #实例化saver
    saver = tf.train.Saver()

    #测试准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Yo, 1), tf.argmax(Yi, 1)),tf.float32))

    with tf.Session() as sess:
        #初始化
        init=tf.global_variables_initializer()
        sess.run(init)

        # #断点续训
        # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess,ckpt.model_checkpoint_path)

        Xt,Yt = (mnist.test.images,mnist.test.labels)
        for i in range(STEPS):
            #训练模型
            Xs,Ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={Xi:Xs,Yi:Ys})

            if  (i > 0  and  i % 500 == 0) or (i == 1) :
                #插入测试
                loss_v = sess.run(loss,feed_dict={Xi:Xt,Yi:Yt})
                acc_v  = sess.run(accuracy,feed_dict={Xi:Xt,Yi:Yt})
                print("1\t After %5d steps, loss on train batch  is \t %f"%(i,loss_value))
                print("2\t                  loss on test  batch  is \t \t %f"%(loss_v))
                print("3\t                  acc  on test  batch  is \t \t \t %f"%(acc_v))

            if i % 2000 == 0 :
                #保存模型
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_SAVE_NAME),global_step=global_step)


def test(mnist_test):
    print("test")
    with tf.Graph().as_default() as g:
        Xt= tf.placeholder(tf.float32,[None,INPUT_NODE])
        Yt= tf.placeholder(tf.float32,[None,OUTPUT_NODE])
        Yo = forward(Xt,REGULARIZER)

        if USE_EMA__ :
            ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            ema_restore = ema.variables_to_restore()
            saver = tf.train.Saver(ema_restore)
        else:
            saver = tf.train.Saver()

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Yo, 1), tf.argmax(Yt, 1)),tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/'[-1]).split('-')[-1]
                    acc=sess.run(accuracy,feed_dict={Xt:mnist_test.test.images,Yt:mnist_test.test.labels})
                    print("1\t After %5d steps, test acc is \t %f"%(global_step,acc))
                else:
                    print('No ckpt file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def pre_pic(picName):
    print("pre_pic")
    img  = Image.open(picName)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50

    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr    = im_arr.reshape([1,784])
    nm_arr    = nm_arr.astype(np.float32)
    ima_ready = np.multiply(nm_arr,1.0/255.0)




def application():

    print("application")

    if 0:
        testNum = input("input the number of the test pictures:")
        for i in range(testNum):
            testPic = raw_input("the path of the pciture:")
            testPicArr = pre_pic(testPic)
            preValue = restore_model(testPicArr)
            print("The prediction number is:",preValue)
    else:
        testPic='./test/5.png'
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is:",preValue)



def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        Xp = tf.placeholder(tf.float32,[None,INPUT_NODE])
        Yp = forward(Xp,REGULARIZER)
        preValue = tf.argmax(Yp,1)


        if USE_EMA__ :
            ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            ema_restore = ema.variables_to_restore()
            saver = tf.train.Saver(ema_restore)
        else:
            saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                preValue = sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print('No ckpt file found')
                return -1



if __name__ == '__main__':




    USE_EMA__ = False

    INPUT_NODE  = 784
    OUTPUT_NODE = 10
    LAYER1_NODE = 500

    BATCH_SIZE  = 200
    STEPS       = 4500

    REGULARIZER               = 0.0001
    LEARNING_RATE_BASE        = 0.1
    LEARNING_RATE_DECAY_STEPS = 1000
    LEARNING_RATE_DECAY_RATE  = 0.99
    MOVING_AVERAGE_DECAY      = 0.99

    MODEL_SAVE_PATH = "./model/"
    MODEL_SAVE_NAME = "mnist_model"

    MAIN_STEP = 2

    if(MAIN_STEP==0):
        mnist = input_data.read_data_sets('./mnist_data/',one_hot=True)

        # # 返回 训练 验证 测试集 子集样本数
        # print("train data size:",mnist.train.num_examples)
        # print("validation data size:",mnist.validation.num_examples)
        # print("test data size:",mnist.test.num_examples)

        # #返回数据和标签
        # print(mnist.train.images[0])
        # print(mnist.train.labels[0])

        backward(mnist)
    elif(MAIN_STEP==1):
        TEST_INTERVAL_SECS = 5
        mnist_test = input_data.read_data_sets('./mnist_data/',one_hot=True)
        test(mnist_test)
    elif(MAIN_STEP==2):
        application()
    else:
        pass



# 留有问题
# 1. 断点续训
# 2. restore 不成功