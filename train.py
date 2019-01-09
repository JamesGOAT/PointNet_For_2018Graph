import argparse

import numpy as np
import tensorflow as tf
import socket
import h5py
import math
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=15360, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=51, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.1, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = ['data/CG/up.h5']
TEST_FILES = ['data/CG/up_test.h5']
OUT_FILES = ['data/CG/up_pred.h5']

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            #print(pointclouds_pl.get_shape().as_list())
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer, epoch)
            eval_one_epoch(sess, ops, test_writer, epoch)
            final_test(sess, ops, epoch)
            fo.write('\n')

            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
        

        #print("Next is the test:\n")
        #final_test()
        
def final_test(sess, ops, num):
    is_training = False
    

    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data = provider.loadDataFile_out(OUT_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        current_label = np.zeros((4,112)) 

        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         #ops['labels_pl']: current_label[start_idx:end_idx][:],
                         ops['is_training_pl']: is_training}
            pred_val = sess.run([ops['pred']], feed_dict=feed_dict)
            x = np.array(pred_val)
            x = np.squeeze(x)
            #if(num > 0 and num % 250 == 0):
                #print(pred_val)
                #fo.write("\nthe "+str(num)+" epoch:\n")     
                #fo.write("\n out for no label for " +str(batch_idx)+" \n")        
                #fo.write("\npred_val:\n")            
                #fo.write(str(x)+'\n')

            
            
            
        



def train_one_epoch(sess, ops, train_writer, num):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    is_training = False
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)


        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx][:],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            if(num % 10 == 0 and batch_idx == 2):
                print(pred_val)
            if(1):
                #fo.write("\nthe "+str(num)+" epoch:\n")    
                #fo.write("\n train for " +str(batch_idx)+" \n")        
                #fo.write("\npred_val:\n")            
                #fo.write(str(pred_val)+'\n')
                #fo.write('\nlabel:\n')
                #fo.write(str(current_label[start_idx:end_idx]))
                #fo.write('\nloss:\n')
                tem = pred_val-current_label[start_idx:end_idx]
                sum = 0.0
                for i in range(112):
                    sum += 1.0 * abs(tem[0][i])
                sum = sum / 112.0
                fo.write(str(sum)+'\n') 
                #print(current_label[start_idx:end_idx][:])          

        
def eval_one_epoch(sess, ops, test_writer, num):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        t_sum = 0.0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx][:],
                         ops['is_training_pl']: is_training}
            summary, step, pred_val = sess.run([ops['merged'], ops['step'],
                 ops['pred']], feed_dict=feed_dict)
            if(1):       
                #fo.write("\nthe "+str(num)+" epoch:\n")   
                #fo.write("\n eval for " +str(batch_idx)+" \n")        
                #fo.write("\npred_val:\n")            
                #fo.write(str(pred_val)+'\n')
                #fo.write('\nlabel:\n')
                #fo.write(str(current_label[start_idx:end_idx]))
                #fo.write('\nloss:\n')
                tem = pred_val-current_label[start_idx:end_idx]
                sum = 0.0
                for i in range(112):
                    sum += 1.0 * abs(tem[0][i])
                sum = sum / 112.0
                t_sum += sum
        t_sum /= num_batches
        print(t_sum)
        
        fo.write(str(t_sum)+'\n')

def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('log/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint("log/"))



if __name__ == "__main__":
    #load_model()
    #!/usr/bin/python
    fo = open("loss_up2.txt", "w")  

    train()
    # 关闭打开的文件
    fo.close()

    #LOG_FOUT.close()
    '''for fn in range(len(TEST_FILES)):
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        print(current_data)'''