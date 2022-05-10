#GPU
import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
%matplotlib inline
USE_GPU = True
if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

# Constant to control how often we print when training models.
print_every = 100
print('Using device: ', device)
######################################1. Barebones TensorFlow：three-layer-convnet######################################
def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    # 用TensorFlow框架写一个三层卷积神经网络
    h1 = tf.nn.relu(tf.nn.conv2d(
        x, conv_w1, strides=[1,1,1,1], padding='SAME', data_format='NHWC'
    )+conv_b1)
    h2 = tf.nn.relu(tf.nn.conv2d(
        h1, conv_w2, strides=[1,1,1,1], padding='SAME', data_format='NHWC'
    )+conv_b2)
    h_fc = flatten(h2)
    scores = tf.matmul(h_fc, fc_w) + fc_b        # Compute scores of shape (N, C)
    return scores
'''
def three_layer_convnet_test():
    
    with tf.device(device):
        x = tf.zeros((64, 32, 32, 3))
        conv_w1 = tf.zeros((5, 5, 3, 6))
        conv_b1 = tf.zeros((6,))
        conv_w2 = tf.zeros((3, 3, 6, 9))
        conv_b2 = tf.zeros((9,))
        fc_w = tf.zeros((32 * 32 * 9, 10))
        fc_b = tf.zeros((10,))
        params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
        scores = three_layer_convnet(x, params)

    # Inputs to convolutional layers are 4-dimensional arrays with shape
    # [batch_size, height, width, channels]
    print('scores_np has shape: ', scores.shape)
'''
#计算损失、梯度、权重更新
def training_step(model_fn, x, y, params, learning_rate):
    with tf.GradientTape() as tape:
        scores = model_fn(x, params) # Forward pass of the model
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        total_loss = tf.reduce_mean(loss)
        grad_params = tape.gradient(total_loss, params)

        # Make a vanilla gradient descent step on all of the model parameters
        # Manually update the weights using assign_sub()
        for w, grad_w in zip(params, grad_params):
            w.assign_sub(learning_rate * grad_w)      
        return total_loss
#定义精度计算函数
def check_accuracy(dset, x, model_fn, params):
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        scores_np = model_fn(x_batch, params).numpy()
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
#定义初始化函数
def create_matrix_with_kaiming_normal(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.keras.backend.random_normal(shape) * np.sqrt(2.0 / fan_in)
def three_layer_convnet_init():
    params = None
    ############################################################################
    # TODO: Initialize the parameters of the three-layer network.              #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    conv_w1 = tf.Variable(create_matrix_with_kaiming_normal((5,5,3,32)))
    conv_b1 = tf.Variable(create_matrix_with_kaiming_normal((1,32)))
    conv_w2 = tf.Variable(create_matrix_with_kaiming_normal((3,3,32,16)))
    conv_b2 = tf.Variable(create_matrix_with_kaiming_normal((1,16)))
    fc_w = tf.Variable(create_matrix_with_kaiming_normal((16 * 32 * 32, 10)))
    fc_b = tf.Variable(create_matrix_with_kaiming_normal((1,10)))
    params = [conv_w1,conv_b1,conv_w2,conv_b2,fc_w,fc_b]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return params
#定义训练函数
def train_part2(model_fn, init_fn, learning_rate):
    '''
    Train a model on CIFAR-10.
    '''
    params = init_fn()  # Initialize the model parameters            
        
    for t, (x_np, y_np) in enumerate(train_dset):
        # Run the graph on a batch of training data.
        loss = training_step(model_fn, x_np, y_np, params, learning_rate)
        
        # Periodically print the loss and check accuracy on the val set.
        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss))
            check_accuracy(val_dset, x_np, model_fn, params)
#开始训练
learning_rate = 3e-3
train_part2(three_layer_convnet, three_layer_convnet_init, learning_rate)

######################################2. Keras Model Subclassing API：three-layer-convnet######################################
class ThreeLayerConvNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2, num_classes):
        super(ThreeLayerConvNet, self).__init__()
        ########################################################################
        # TODO: Implement the __init__ method for a three-layer ConvNet. You   #
        # should instantiate layer objects to be used in the forward pass.     #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****      
        initializerConv = tf.initializers.VarianceScaling(scale=2.0)  #scale小数点后几位，精度？
        self.conv1 = tf.keras.layers.Conv2D(filters=channel_1, kernel_size=(5,5), strides=(1, 1), padding='same', use_bias=True, bias_initializer='zeros',
                                       activation='relu', kernel_initializer=initializerConv)
        self.conv2 = tf.keras.layers.Conv2D(filters=channel_2, kernel_size=(3,3), strides=(1, 1), padding='same', use_bias=True, bias_initializer='zeros',
                                       activation='relu', kernel_initializer=initializerConv)

        self.flatten1 = tf.keras.layers.Flatten()                                  

        self.fc1 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=initializerConv)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
    def call(self, x, training=False):
        scores = None
        ########################################################################
        # TODO: Implement the forward pass for a three-layer ConvNet. You      #
        # should use the layer objects defined in the __init__ method.         #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten1(x)
        scores = self.fc1(x)
       
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return scores
'''
def test_ThreeLayerConvNet():    
    channel_1, channel_2, num_classes = 12, 8, 10
    model = ThreeLayerConvNet(channel_1, channel_2, num_classes)
    with tf.device(device):
        x = tf.zeros((64, 3, 32, 32))
        scores = model(x)
        print(scores.shape)
'''
#训练函数
def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1, is_training=False):
    with tf.device(device):

        # Compute the loss like we did in Part II
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        model = model_init_fn()
        optimizer = optimizer_init_fn()
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        t = 0
        for epoch in range(num_epochs):
            
            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            
            for x_np, y_np in train_dset:
                with tf.GradientTape() as tape:
                    
                    # Use the model function to build the forward pass.
                    scores = model(x_np, training=is_training)
                    loss = loss_fn(y_np, scores)
      
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    # Update the metrics
                    train_loss.update_state(loss)
                    train_accuracy.update_state(y_np, scores)
                    
                    if t % print_every == 0:
                        val_loss.reset_states()
                        val_accuracy.reset_states()
                        for test_x, test_y in val_dset:
                            # During validation at end of epoch, training set to False
                            prediction = model(test_x, training=False)
                            t_loss = loss_fn(test_y, prediction)

                            val_loss.update_state(t_loss)
                            val_accuracy.update_state(test_y, prediction)
                        
                        template = 'Iteration {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
                        print (template.format(t, epoch+1,
                                             train_loss.result(),
                                             train_accuracy.result()*100,
                                             val_loss.result(),
                                             val_accuracy.result()*100))
                    t += 1

#初始化网络参数
def model_init_fn():
    model = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return ThreeLayerConvNet(channel_1, channel_2, num_classes)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return model
learning_rate = 3e-3
#初始化优化函数
def optimizer_init_fn():
    optimizer = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer
#开始训练网络
channel_1, channel_2, num_classes = 32, 16, 10
train_part34(model_init_fn, optimizer_init_fn)

######################################3. Keras Sequential API：three-layer-convnet######################################
def model_init_fn():
    model = None
    ############################################################################
    # TODO: Construct a three-layer ConvNet using tf.keras.Sequential.         #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    input_shape = (32, 32, 3)
    channel_1, channel_2, num_classes = 32, 16, 10
    initializerConv = tf.initializers.VarianceScaling(scale=2.0)
    layers = [
        tf.keras.layers.Conv2D(filters=channel_1, kernel_size=(5,5), strides=(1, 1), padding='same', use_bias=True, bias_initializer='zeros',
                                       activation='relu', kernel_initializer=initializerConv),
        tf.keras.layers.Conv2D(filters=channel_2, kernel_size=(3,3), strides=(1, 1), padding='same', use_bias=True, bias_initializer='zeros',
                                       activation='relu', kernel_initializer=initializerConv),
        tf.keras.layers.Flatten(),                                  

        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=initializerConv),
    ]
    model = tf.keras.Sequential(layers)
    return model

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################

learning_rate = 5e-4
def optimizer_init_fn():
    optimizer = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return tf.keras.optimizers.SGD(learning_rate=learning_rate) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer

train_part34(model_init_fn, optimizer_init_fn)
#away train loop
model = model_init_fn()
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)



