import tensorflow as tf
import pandas as pd
import tensorlayer as tl
import numpy as np

def readcsv(name):
    inp = pd.read_csv(name,sep=',',header=None)
    arr =  inp.as_matrix()
    feed = []
    tmp = []
    cut = 1
    for i in arr:
        if(cut != 12):
            cut = cut+1
            tmp.append(i[1])
        else:
            tmp.append(i[1])
            tmp.append(i[0])
            feed.append(tmp)
            cut = 1
            tmp = []
    return feed

def makefeed(feed):
    res = []
    for i in feed:
        res.append(i[0:12])
    ress = np.asarray(res)
    return ress

def getval(feed):
    res=[]
    for i in feed:
        res.append(int(i[12]-1))
    ress = np.asarray(res)
    return ress

input_name = "ninput.csv"
feed = readcsv(input_name)

prcp_feed = makefeed(feed)
prcp_val = getval(feed)


ta_name = "nval.csv"
ta_feed = readcsv(ta_name)

t_feed = makefeed(ta_feed)
t_val = getval(ta_feed)

batch_size = 10
epoch = 0
for i in feed:
    epoch = epoch+1

log_dir = 'log'

sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,shape=(None,12),name = 'x_input')
    v = tf.placeholder(tf.int64,shape=(None),name='val')

network = tl.layers.InputLayer(x,name='input_layer')
network = tl.layers.BatchNormLayer(network,name='batchnorm1')
network = tl.layers.DenseLayer(network,n_units=24,act=tf.nn.relu,name='relu1',b_init=1.0,W_init=tf.random_uniform_initializer(minval=0,maxval=1,seed=1))
network = tl.layers.DropoutLayer(network,keep=0.9,name='dropout')
network = tl.layers.DenseLayer(network,n_units=24,act=tf.nn.tanh,name='tanh',b_init=1.0,W_init=tf.random_uniform_initializer(minval=0,maxval=1,seed=1))
network = tl.layers.DenseLayer(network,n_units=24,act=tf.nn.relu,name='relu2',b_init=1.0,W_init=tf.random_uniform_initializer(minval=0,maxval=1,seed=1))
network = tl.layers.BatchNormLayer(network,name='batchnorm2')
network = tl.layers.DenseLayer(network,n_units=5,name='output_layer')

with tf.name_scope('loss'):
    y = network.outputs
    tf.summary.histogram('output',y)
    cost = tl.cost.cross_entropy(y,v,name='cost')
    correct_pred = tf.equal(tf.argmax(y,1),v)
    acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    tf.summary.scalar('cost',cost)
    tf.summary.scalar('classific error',acc)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.035, global_step, decay_steps=100, decay_rate=0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

with tf.name_scope('learning_rate'):
    tf.summary.scalar('learning_rate',learning_rate)

tl.layers.initialize_global_variables(sess)

network.print_layers()
network.print_params()

saver = tf.train.Saver()
save_path = 'prcp_save/model.ckpt'

tl.utils.fit(sess,network,train_step,cost,prcp_feed,prcp_val,x,v,acc=acc,batch_size=10,n_epoch=100,print_freq=10,X_val=t_feed,y_val=t_val,eval_train=False,tensorboard=True)

saver.save(sess,save_path)
sess.close()