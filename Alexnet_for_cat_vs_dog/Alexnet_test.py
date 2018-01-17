import os
import numpy as np
import tensorflow as tf
import csv


def get_test_files(file_dir):
    test_picture = []
    test_lable = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        test_picture.append(file_dir + file)
        test_lable.append(name[0])
    temp = np.array([test_picture, test_lable])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

def get_test_batch(image,label,image_W, image_H,batch_size,capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)
    #label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch,label_batch

def inference(images,batch_size,n_classes):
    #  input images 4D tensor (batch_size,width,height,channels)  tf.float32
    #        batch_size int32
    #        n_classes  int32
    #  ALex input 227*227*3
    # conv1
    with tf.variable_scope('conv1') as scope:
        weight = tf.get_variable('weight1',              # conv1  kernal size 11*11*96
                                 shape=[11,11,3,96],     # initialize means:0 stddev:0.01
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias1',
                               shape=[96],               #bias 96
                               dtype=tf.float32,         #initialize 0
                               initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images,weight,strides=[1,4,4,1],padding='VALID')    #conv
        pre_activation = tf.nn.bias_add(conv,bias)                              # add bias
        conv1 = tf.nn.relu(pre_activation,name=scope.name)                      #relu activation
    #pool1
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1,                    #pool1  maxpool  kernal size 3*3
                               ksize=[1,3,3,1],              #stride 2   pad 0
                               strides=[1,2,2,1],
                               padding='VALID',
                               name='pool1')
    #norm1
    with tf.variable_scope('norm1') as scope:                    #norm1 : LRN
        norm1 = tf.nn.lrn(pool1,
                          depth_radius=4,
                          bias=1.0,
                          alpha=0.01/9.0,
                          beta=0.75,
                          name='norm1')
    #conv2
    with tf.variable_scope('conv2') as scope:                        #conv2 5*5*256
        weight=tf.get_variable('weight2',                   #stride 1  padding 2
                               shape=[5,5,96,256],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias2',                     #bias 256   init 1
                               shape=[256],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(norm1,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv2 = tf.nn.relu(pre_activation,name=scope.name)
    #pool2
    with tf.variable_scope('pool2') as scope:                     #pool2 ksize : 3*3   stride: 2   padding :0
        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1,3,3,1],
                               strides=[1,2,2,1],
                               padding='VALID',
                               name='pool2')
    #norm2
    with tf.variable_scope('norm2') as scope:                    #norm2:LRN
        norm2 = tf.nn.lrn(pool2,
                          depth_radius=4,
                          bias=1.0,
                          alpha=0.01/9.0,
                          beta=0.75,
                          name='norm2')
    #conv3
    with tf.variable_scope('conv3') as scope:
        weight = tf.get_variable('weight3',                 #conv3:   size:3*3*384  stride:1  pad:1
                                 shape=[3,3,256,384],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias3',                         #bias3:   size:384  init 0
                               shape=[384],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(norm2,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv3 = tf.nn.relu(pre_activation)
    #conv4
    with tf.variable_scope('conv4') as scope:
        weight=tf.get_variable('weight4',             #conv4  size:3*3*384  stride:1 pad:1
                               shape=[3,3,384,384],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias4',
                               shape=[384],             #bias4  size 384  init 1
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(conv3,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv4 = tf.nn.relu(pre_activation)
    #conv5
    with tf.variable_scope('conv5') as scope:
        weight = tf.get_variable('weight5',          #conv5   3*3*256  stride 1 pad 1
                                 shape=[3,3,384,256],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias5',
                               shape=[256],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        conv = tf.nn.conv2d(conv4,weight,strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        conv5 = tf.nn.relu(pre_activation)
    #pool5
    with tf.variable_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1,3,3,1],
                               strides=[1,1,1,1],
                               padding='VALID',
                               name='pool5')
    #fc6
    with tf.variable_scope('full_connect6') as scope:
        reshape = tf.reshape(pool5, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight = tf.get_variable(name='weight6',
                                 shape=[dim,4096],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias6',
                               shape=[4096],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        pre_activation = tf.matmul(reshape,weight)+bias
        pre_dropout = tf.nn.relu(pre_activation,name=scope.name)
        fc6 = tf.nn.dropout(pre_dropout,keep_prob=0.5)
    #fc7
    with tf.variable_scope('full_connect7') as scope:
        weight = tf.get_variable('weight7',
                                 shape=[4096,4096],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias7',
                               shape=[4096],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        pre_activation = tf.matmul(fc6,weight)+bias
        pre_dropout = tf.nn.relu(pre_activation,name=scope.name)
        fc7 = tf.nn.dropout(pre_dropout,keep_prob=0.5)
    #fc8
    with tf.variable_scope('full_connect8') as scope:
        weight = tf.get_variable('weight8',
                                 shape=[4096,n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        bias = tf.get_variable('bias8',
                               shape=[n_classes],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(1))
        fc8 = tf.add(tf.matmul(fc7,weight),bias,name='fc8')
    return fc8

def predict_one_image():
    test_dir = '/home/work/test/test/'
    logs_train_dir = '/home/work/log2/'
    test,test_label = get_test_files(test_dir)

    test_batch,label_batch  = get_test_batch(test,test_label,227,227,1,2000)
    test_logits = inference(test_batch, 1, 2)
    logit = tf.nn.softmax(test_logits)
    label_visual = label_batch
    #train_loss = model.losses(train_logits, train_label_batch)
    #train_op = model.trainning(train_loss, learning_rate)
    #train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    #train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image_list = []
    label_list = []
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')

    for step in np.arange(12500):
        if coord.should_stop():
                break
        if step%500==0:
            print(step)
            #print("Reading checkpoints...")
            #ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            #if ckpt and ckpt.model_checkpoint_path:
                #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                #saver.restore(sess, ckpt.model_checkpoint_path)
                #print('Loading success, global_step is %s' % global_step)
            #else:
                #print('No checkpoint file found')
        prediction,label = sess.run([logit,label_visual])
            #prediction = sess.run(test_logits)
            #max_index = np.argmax(prediction)
        dog_prob = prediction[0][1]
            #label_batch = sess.run(label_visual)
            #print(label)
            #print(prediction)
            #print(max_index)
        image_list.append(label)
        label_list.append(dog_prob)

    coord.request_stop()

    coord.join(threads)
    sess.close()
    return image_list,label_list

#predict_one_image()
#print('predic done')
#run_training()

a,b = predict_one_image()
csvfile = open('/home/work/log2/sample_submission.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerow(['id','label'])
for i in range(12500):
    print (i)
    writer.writerow([a[i][0],b[i]])
csvfile.close()
print('done')
