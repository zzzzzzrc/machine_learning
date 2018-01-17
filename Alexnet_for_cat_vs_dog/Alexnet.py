import tensorflow as tf
import tensorflow as tf
import numpy as np
import os

#%%

# you need to change this to your data directory
#train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'


def get_test_files(file_dir):
    test_picture = []
    test_lable = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        test_picture.append(file_dir + file)
        test_lable.append(name[0])
    temp = np.array([test_picture, test_lable])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]


    return image_list, label_list


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    image_list = np.hstack((cats,dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]


    return image_list, label_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    #image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)

    #you can also use shuffle_batch
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch
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

def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def training(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):

    with tf.variable_scope('accuracy') as scope:
      #correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.equal(tf.argmax(logits,1),labels)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
N_CLASSES = 2
IMG_W = 227  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 227
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 30000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001

def run_training():

    # you need to change the directories to yours.
    train_dir = '/home/work/train/'
    logs_train_dir = '/home/work/log2/'

    train, train_label = get_files(train_dir)

    train_batch, train_label_batch = get_batch(train,
                                                train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = losses(train_logits, train_label_batch)
    train_op = training(train_loss, learning_rate)
    train__acc = evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
            #logits = sess.run(train_logits)
            #logits_new =  tf.nn.softmax(logits)
            #pre = sess.run(logits_new)

            if step % 200 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
run_training()


def predict_one_image():
    test_dir = 'D:\\machine_learning\\dogs_vs_cats\\test\\test\\'
    logs_train_dir = 'D:\\machine_learning\\dogs_vs_cats\\log\\'
    test,test_label = get_test_files(test_dir)

    test_batch,label_batch  = get_batch(test,test_label,208,208,1,2000)
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
    try:
        for step in np.arange(12500):
            if coord.should_stop():
                    break
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
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    return image_list,label_list

#predict_one_image()

#run_training()

'''a,b = predict_one_image()
csvfile = open('D:\\machine_learning\\dogs_vs_cats\\sample_Submission.csv', 'w',newline='')
writer = csv.writer(csvfile)
writer.writerow(['id','label'])
for i in range(12500):
    writer.writerow([a[i][0],b[i]])
csvfile.close()'''
