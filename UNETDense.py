#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
from __future__ import division
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import skimage.io as io
import glob
from matplotlib import pyplot as plt
class DataGenerator:
    train_path='drive/train'
    path='drive/train/ori'
    val_path='drive/validation/'
        
    seed=1
    image_save_prefix  = "image"
    mask_save_prefix  = "mask"
    mask_color_mode = "grayscale"
    save_to_dir = None
    image_color_mode = "grayscale"
    ori_dir=path+'/*'
    bi_dir=path+'_bi/*'
    image_folder='images'
    mask_folder='masks'
    target_size = (256,256)

    def __init__(self,num_slice,data_gen_args,batch_size):
    
        self.num_slice=num_slice
        self.data_gen_args=data_gen_args
        self.batch_size=batch_size                
    def adjustData(self,img,mask):   
        if(np.max(img) > 1):
            img = img / 255.0
        if (np.max(mask)>1):
            mask=mask/255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        mask=mask.astype(int)
        return (img,mask)
    def random_slice_data(self):
        a=glob.glob(self.ori_dir)
        count1=0
        count2=0
        count3=0
        for i in a:
            count3+=1
            impath=i
            bipath=self.path+'_bi'+i[31:-4]+'a.png'
            ori=io.imread(impath,as_gray=True)
            bi=io.imread(bipath,as_gray=True)
            ori,bi=self.adjustData(ori,bi)
            assert ori.shape==bi.shape
            s=ori.shape
            xi=np.random.permutation(s[1]-256)
            yi=np.random.permutation(s[1]-256)
            for j in range(self.num_slice):
                if count3%5==1:
                    count1+=1
                    plt.imsave( 'drive/validation/images/'+str(count1)+'.png',ori[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')
                    plt.imsave( 'drive/validation/masks/'+str(count1)+'.png',bi[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')
                else:
                    count2+=1
                    plt.imsave( 'drive/train/images/'+str(count2)+'.png',ori[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')
                    plt.imsave( 'drive/train/masks/'+str(count2)+'.png',bi[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')

    def trainGenerator(self):

        image_datagen = ImageDataGenerator(**self.data_gen_args)
        mask_datagen = ImageDataGenerator(**self.data_gen_args)
        
        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes = [self.image_folder],
            class_mode = None,
            color_mode = self.image_color_mode,
            target_size = self.target_size,
            batch_size = self.batch_size,
            save_to_dir = self.save_to_dir,
            save_prefix  = self.image_save_prefix,
            seed =self.seed)
        mask_generator = mask_datagen.flow_from_directory(
            self.train_path,
            classes = [self.mask_folder],
            class_mode = None,
            color_mode = self.mask_color_mode,
            target_size = self.target_size,
            batch_size = self.batch_size,
            save_to_dir = self.save_to_dir,
            save_prefix  = self.mask_save_prefix,
            seed =self.seed)
        train_generator = zip(image_generator, mask_generator)
        for (img,mask) in train_generator:
            img,mask = self.adjustData(img,mask)
            yield (img,mask)  
    def valData(self):
        #print(io.ImageCollection(self.val_path+self.image_folder+'/*',load_func=self.imread_convert))
        image_collection =np.expand_dims(np.array(io.ImageCollection(self.val_path+self.image_folder+'/*',load_func=self.imread_convert)),3)
        mask_collection=np.expand_dims(np.array(io.ImageCollection(self.val_path+self.mask_folder+'/*',load_func=self.imread_convert)),3)
        #print(image_collection.shape)
        print(mask_collection.shape)
        val_data=self.adjustData(image_collection,mask_collection)
        return val_data
    def imread_convert(self,f):
        return io.imread(f,as_gray=True)
        
data_gen_args = dict(rotation_range=0.2,
                    shear_range=0.05,
                    zoom_range=0.05,
                    vertical_flip=True,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    )


# In[3]:


'''Dense Unet'''
from keras import backend as K
from keras import backend 
from keras.models import Model
from keras import models
from keras import layers
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, concatenate, Concatenate, UpSampling2D, Activation
from keras.losses import binary_crossentropy
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from functools import partial
from keras.layers import Lambda
from keras.applications.densenet import DenseNet121
import keras
import numpy as np
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import applications
bn_axis = 3
channel_axis = bn_axis

def _generate_layer_name(name, branch_idx=None, prefix=None):
    """Utility function for generating layer names.
    If `prefix` is `None`, returns `None` to use default automatic layer names.
    Otherwise, the returned layer name is:
        - PREFIX_NAME if `branch_idx` is not given.
        - PREFIX_Branch_0_NAME if e.g. `branch_idx=0` is given.
    # Arguments
        name: base layer name string, e.g. `'Concatenate'` or `'Conv2d_1x1'`.
        branch_idx: an `int`. If given, will add e.g. `'Branch_0'`
            after `prefix` and in front of `name` in order to identify
            layers in the same block but in different branches.
        prefix: string prefix that will be added in front of `name` to make
            all layer names unique (e.g. which block this layer belongs to).
    # Returns
        The layer name.
    """
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))

def conv2d_bn(x,
              filters,
              kernel_size,
              
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=False,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = _generate_layer_name('BatchNorm', prefix=name)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = _generate_layer_name('Activation', prefix=name)
        x = Activation(activation, name=ac_name)(x)
    return x



        

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def softmax_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.5 + dice_coef_loss(y_true, y_pred) * 0.5# + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2

def dice_coef_rounded_ch0(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def conv_block1(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x
def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = BatchNormalization(name=name, axis=bn_axis)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv
def dense_block(x, blocks, name):

    for i in range(blocks):
        x = conv_block1(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x
def get_dense_unet_softmax(input_shape, weights='imagenet'):
    inp = Input(input_shape + (1,))
    blocks = [6, 12, 24, 16]
    
    
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inp)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    conv1 = x
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    conv2 = x
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    conv3 = x
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    conv4 = x
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)
    conv5 = x 
    
    conv6 = conv_block(UpSampling2D()(conv5), 320)
    conv6 = concatenate([conv6, conv4], axis=-1)
    conv6 = conv_block(conv6, 320)

    conv7 = conv_block(UpSampling2D()(conv6), 256)
    conv7 = concatenate([conv7, conv3], axis=-1)
    conv7 = conv_block(conv7, 256)

    conv8 = conv_block(UpSampling2D()(conv7), 128)
    conv8 = concatenate([conv8, conv2], axis=-1)
    conv8 = conv_block(conv8, 128)

    conv9 = conv_block(UpSampling2D()(conv8), 96)
    conv9 = concatenate([conv9, conv1], axis=-1)
    conv9 = conv_block(conv9, 96)

    conv10 = conv_block(UpSampling2D()(conv9), 64)
    conv10 = conv_block(conv10, 64)
    res = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    res = BatchNormalization()(res)
    res = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(res)
   
    model = Model(inp, res)
    
    if weights == 'imagenet':
        densenet = DenseNet121(input_shape=input_shape + (3,), weights=weights, include_top=False)
        w0 = np.array(densenet.layers[2].get_weights())
        w = np.array(model.layers[2].get_weights())
        w[0][:, :, 0, :] = w0[0][:, :, 0, :]

        model.layers[2].set_weights(w)
        for i in range(3, len(densenet.layers)):
            
            model.layers[i].set_weights(densenet.layers[i].get_weights())
            model.layers[i].trainable = True
    print(model.summary())
    model.save('00.hdf5')
    return model

print('training model ...')
filepath="{epoch:02d}-{val_loss:.4f}.hdf5"
modelcheckpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto',period=10)

input_shape=(256,256)
model=get_dense_unet_softmax(input_shape)


myGenerator = DataGenerator(200,data_gen_args,16)
train_generator=myGenerator.trainGenerator()
val_data=myGenerator.valData()
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-7)
 
model.compile(loss=softmax_dice_loss,
              optimizer=Adam(lr=1e-4, amsgrad=True),metrics=[dice_coef_loss,  keras.metrics.binary_crossentropy])

model_info=model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=100,verbose=1,callbacks=[modelcheckpoint,reduce_lr],validation_data=val_data)

import matplotlib.pyplot as plt
def plot_model_history(model_history):
    # summarize history for loss
    plt.plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    plt.plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.savefig('LOSS.png')
    plt.show()
plot_model_history(model_info)


# In[ ]:


from keras.losses import binary_crossentropy
from keras import backend as K
from keras import backend 
from keras.models import Model
from keras import models
from keras import layers
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, concatenate, Concatenate, UpSampling2D, Activation
from keras.losses import binary_crossentropy
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from functools import partial
from keras.layers import Lambda
from keras.applications.densenet import DenseNet121
import keras
import numpy as np
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import applications
modelcheckpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto',period=5)
def focal_loss(y_true, y_pred):
   gamma =2.0
   alpha = 0.25
   pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
   pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
   return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def softmax_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.5 + dice_coef_loss(y_true, y_pred) * 0.5# + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2

from keras.models import load_model
model = load_model('10-0.0148.hdf5',custom_objects={'dice_coef_loss':dice_coef_loss,'softmax_dice_loss':softmax_dice_loss})

densenet = DenseNet121(input_shape=input_shape + (3,), include_top=False)
for i in range(2, len(densenet.layers)):
    model.layers[i].trainable = True
print(model.summary())
filepath="{epoch:02d}-{val_loss:.4f}.hdf5"
modelcheckpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto',period=5)

input_shape=(256,256)
myGenerator = DataGenerator(200,data_gen_args,16)
train_generator=myGenerator.trainGenerator()
val_data=myGenerator.valData()
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-7)
model.compile(loss=softmax_dice_loss,
              optimizer=Adam(lr=1e-5, amsgrad=True), metrics=[dice_coef_loss,  keras.metrics.binary_crossentropy])

model_info=model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=100,verbose=1,callbacks=[modelcheckpoint,reduce_lr],validation_data=val_data)

import matplotlib.pyplot as plt
def plot_model_history(model_history):
    # summarize history for loss
    plt.plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    plt.plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.savefig('LOSS.png')
    plt.show()
plot_model_history(model_info)


