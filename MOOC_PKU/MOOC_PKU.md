# 来源
bilibili视频:  
[【tensorflow】基础教学课程](https://www.bilibili.com/video/av25541757)   

icourse163：   
[人工智能实践：Tensorflow笔记](https://www.icourse163.org/course/PKU-1002536002)    



# 3.1 张量 计算图 会话
张量：多维数组（列表）
阶： 张量的维数
数据类型：tf.float32 tf.int32 ...

# 3.2 前向传播
## 参数： 线上的权重
```
w1=tf.Variable(tf.random_normal([2,3],stddev=2,mean=0,seed=1))
w2=tf.Variable(tf.truncated_normal([2,3],stddev=2,mean=0,seed=1))
w3=tf.Variable(tf.random_uniform([2,3],minval=0,maxval=1))
```

w1,正态分布，2x3矩阵 标准差为2 均值为0 随机种子为1  
w2,去掉过大偏离点的正态分布  
w3,在均匀分布中随机采样

# 3.3 反向传播
反向传播：优化模型参数，在所有参数上用梯度下降，使NN模型在训练数据上的损失函数最小。  

loss： 预测值与已知答案的差距  

均方误差MSE：y_和y的差的平方的平均数
`loss=tf.reduce_mean(tf.square(y_-y))`

反向传播训练方法
```
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
train_step = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss)
```

学习率：决定参数每次更新的幅度


# 4.1 激活函数 损失函数

```
relu      fx=max(x,0)
sigmoid   fx=1/(1+e-x)
tanh      fx=(1-e-2x)/(1+e-2x)

loss = tf.reduce_mean(tf.square(y_-y))
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),COST*(y-y_),PROFIT*(y_-y)))
```


eq.销量与x1x2有关 y_=x1+x2 噪声-0.05~+0.05

## 交叉熵
H(y_.y) = - ∑ y_ * logy
ce = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-12,1.0)))
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
cem=tf.reduce_mean(ce)

# 4.2 学习率
指数衰减学习率
```

LEARNING_RATE_BASE        = 0.1
LEARNING_RATE_DECAY_STEPS = 1
LEARNING_RATE_DECAY_RATE  = 0.99
global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_DECAY_STEPS,LEARNING_RATE_DECAY_RATE,staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
```

# 4.3 滑动平均

影子     = 衰减率*影子 + （1-衰减率)*参数
影子初值 = 参数初值
衰减率   = min{MOVING_AVERAGE_DECAY, 1+轮数/10+轮数}
```
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
```

ema.apply()函数实现对括号内参数求滑动平均，tf.trainable_variables()函数实现把所有
待训练参数汇总为列表。
```
ema_op = ema.apply(tf.trainable_variables()) 
```


tf.control_dependencies函数实现将滑动平均和训练过程同步运行。
```
with tf.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name='train') 
```

查看模型中参数的平均值，可以用 ema.average()函数。
```
sess.run(ema.average(w1)
```

# 过拟合 正则化
√过拟合：神经网络模型在训练数据集上的准确率较高，在新的数据进行预测或分类时准确率较
低，说明模型的泛化能力差。
√正则化：在损失函数中给每个参数 w 加上权重，引入模型复杂度指标，从而抑制模型噪声，减小
过拟合。 


使用正则化后，损失函数 loss 变为两项之和：
```
loss = loss_orign + REGULARIZER*loss(w)
```

loss(w)有两种 l1与l2
① L1 正则化： 𝒍𝒐𝒔𝒔𝑳𝟏 = ∑𝒊 |𝒘𝒊 |
`loss(w) = tf.contrib.layers.l1_regularizer(REGULARIZER)(w)`
② L2 正则化： 𝒍𝒐𝒔𝒔𝑳𝟐 = ∑𝒊 |𝒘𝒊 | 𝟐 
`loss(w) = tf.contrib.layers.l2_regularizer(REGULARIZER)(w)`


## eq 随机点，  x0^2+x1^2<2时 为1类 其他2类
```
import matplotlib.pyplot as plt
plt.scatter(x,y,c="颜色")
plt.show()
```

xx,yy=np.mgrid[-5:5:0.25,-5:5:0.25]
grid = np.c_[xx.ravel(),yy.ravel()]




# 基本操作
## 模型的保存与加载

### 保存
在反向传播过程中，一般会间隔一定轮数保存一次神经网络模型，并产生三个文件（保存当前图结构的.meta 文件、保存当前参数名的.index 文件、保存当前参数的.data 文件），在 Tensorflow 中如下表示：
```
saver = tf.train.Saver()          
with tf.Session() as sess:    
    for i in range(STEPS): 
        if i % 轮数 == 0:          
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step)
```
### 加载

若 ckpt 和保存的模型在指定路径中存在，则将保存的神经网络模型加载到当前会话中。
```
with tf.Session() as sess: 
    ckpt = tf.train.get_checkpoint_state(存储路径) 
    if ckpt and ckpt.model_checkpoint_path: 
      saver.restore(sess, ckpt.model_checkpoint_path) 
```

### 加载滑动平均
在保存模型时，若模型中采用滑动平均，则参数的滑动平均值会保存在相应文件中。通过实例化 saver 对象，实现参数滑动平均值的加载，在 Tensorflow 中如下表示
```
ema = tf.train.ExponentialMovingAverage(滑动平均基数) 
ema_restore = ema.variables_to_restore()         
saver = tf.train.Saver(ema_restore) 
```

# mnist 神经网络模型准确率评估方法 
在网络评估时，一般通过计算在一组数据上的识别准确率，评估神经网络的效果。
```
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
```




# 一些函数

tf.get_collection("")             | 从 collection 集合中取出全部变量生成一个列表
tf.add()                          | 将参数列表中对应元素相加
tf.convert_to_tensor              | 转换为tf数据格式
tf.cast(x,dtype)                  | 将参数 x 转换为指定数据类型
tf.equal()                        | 对比两个矩阵或者向量的元素。若对应元素相等，则返 回 True；若对应元素不相等，则返回 False。
tf.reduce_mean(x,axis)            | 求取矩阵或张量指定维度的平均值。若不指定第二个参数，则在所有元素中取平均值；若指定第二个参数为 0，则在第一维元素上取平均值，即每一列求平均值；若指定第二个参数为 1，则在第二维元素上取平均值，即每一行求平均值。
tf.argmax(x,axis)                 | 返回指定维度 axis 下，参数 x 中最大值索引号。
with tf.Graph().as_default() as g | 将当前图设置成为默认图，并返回一个上下文管理器。应用于将已经定义好的神经网络在计算图中复现。



os.path.join('/hello/','good/boy/','doiido')             | 拼接路径
'./model/mnist_model-1001'.split('/')[-1].split('-')[-1] |拆分路径


```
A = tf.convert_to_tensor(np.array([[1,1,2,4], [3,4,8,5]]))   
print A.dtype   
b = tf.cast(A, tf.float32)   
print b.dtype 
```
输出结果：  
<dtype: 'int64'> 
<dtype: 'float32'> 

## tf.equal()
```
A = [[1,3,4,5,6]]   
B = [[1,3,4,3,2]]   
with tf.Session() as sess:   
    print(sess.run(tf.equal(A, B))) 
```
输出结果：[[ True  True  True False False]]


## tf.reduce_mean(x,axis)
```
x = [[1., 1.] [2., 2.]] 
print(tf.reduce_mean(x)) 
print(tf.reduce_mean(x, 0)) 
print(tf.reduce_mean(x, 1)) 
```
输出结果：
1.5    
[1.5, 1.5]    
[1., 1.]    

## tf.argmax(x,axis)
tf.argmax([1,0,0],1)
函数中，axis 为 1，参数 x 为[1,0,0]，表示在参数 x
的第一个维度取最大值对应的索引号，故返回 0。 