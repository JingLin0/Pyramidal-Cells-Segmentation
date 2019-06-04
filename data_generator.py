# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:12:43 2019

@author: Administrator
"""
from __future__ import print_function
from __future__ import division
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import skimage.io as io
import glob
from matplotlib import pyplot as plt
class DataGenerator:
    train_path='E:/pyramidal/jingCode/train'
    path='E:/pyramidal/jingCode/train/ori'
    val_path='E:/pyramidal/jingCode/validation/'
        
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
                    plt.imsave( 'E:/pyramidal/jingCode/validation/images/'+str(count1)+'.png',ori[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')
                    plt.imsave( 'E:/pyramidal/jingCode/validation/masks/'+str(count1)+'.png',bi[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')
                else:
                    count2+=1
                    plt.imsave( 'E:/pyramidal/jingCode/train/images/'+str(count2)+'.png',ori[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')
                    plt.imsave( 'E:/pyramidal/jingCode/train/masks/'+str(count2)+'.png',bi[xi[j]:xi[j]+256,yi[j]:yi[j]+256],cmap='gray')

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
            classes = [self.image_folder],
            class_mode = None,
            color_mode = self.image_color_mode,
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
        
        image_collection =np.expand_dims(np.array(io.ImageCollection(self.val_path+self.image_folder+'/*',load_func=self.imread_convert)),3)
        mask_collection=np.expand_dims(np.array(io.ImageCollection(self.val_path+self.mask_folder+'/*',load_func=self.imread_convert)),3)
        print(image_collection.shape)
        val_data=self.adjustData(image_collection,mask_collection)
        return val_data
    def imread_convert(self,f):
        return io.imread(f,as_gray=True)
        
data_gen_args = dict(rotation_range=20,
                    shear_range=0.15,
                    zoom_range=0.15,
                    vertical_flip=True,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    brightness_range=[0.5, 1.5])
#%%
#
#myGenerator = DataGenerator(20,data_gen_args,32)#"E:\\pyramidal\\jingCode\\train\\aug"
#myGenerator.random_slice_data()
#val_data=myGenerator.valData()
#print(len(val_data))
##%%
#m=myGenerator.trainGenerator()
#num_batch = 5
#for i,batch in enumerate(m):
#    print(batch[1].shape)
#    if(i >= num_batch):
#        break
#%%


#def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
#    for i in range(num_image):
#        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
#        img = img / 255
#        img = trans.resize(img,target_size)
#        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#        img = np.reshape(img,(1,)+img.shape)
#        yield img

    #%%
    print(0%5)