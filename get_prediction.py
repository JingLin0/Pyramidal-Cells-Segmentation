# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:17:22 2019

@author: Administrator
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import glob
import skimage.io as io

def slice_data(image,row,col,size_image):
    '''slice whole images into (row*col)*256*256 sub images '''
    image_array=[]
    for i in range(row):
        for j in range(col):
            image_array.append(image[i*size_image:(i+1)*size_image,j*size_image:(j+1)*size_image])
    image_array=np.array(image_array)
   
    return image_array


def read_images(path,size_image):
    all_images_array=[]
    all_images_path=glob.glob(path)
    num_images=len(all_images_path)
    for i in range(num_images):
        im=io.imread(path[:-1]+'ori('+str(i+1)+').png')
        print(im.shape)
        m,n=im.shape
        row=np.floor(m/size_image).astype(int)
        col=np.floor(n/size_image).astype(int)
        all_images_array.append(slice_data(im,row,col,size_image))
    all_images_array=np.array(all_images_array)
    all_images_array=np.expand_dims(np.concatenate(all_images_array,axis=0),axis=3)
    np.save('E:/pyramidal/jingCode/test',all_images_array)

    return row,col,num_images, all_images_array 
def reconstruct(row,col,num_image,mat):
    mat=np.squeeze(mat)
    all_image=[]
    for i in range(num_image):
        image=[]
        for j in range(row):
            image.append(np.concatenate(mat[j*col:(j+1)*col,:],axis=1))
        image=np.array(image)
        image=np.concatenate(image,0)
        print(image.shape)
        all_image.append(image)
    all_image=np.array(all_image)
    print(all_image.shape)
    return all_image

path='E:/pyramidal/jingCode/099_00_800nm_newScanner_Part1_code_data/099_00_800nm_newScanner_Part1/*'
size_image=256
row,col,num_images,all_sub_images=read_images(path,size_image)
print('We have %d images in total.'%num_images)
print('Each image is split into %d rows and %d columns sub images with size 256*256.'%(row,col))
print(all_sub_images.shape)
#%%

mat = np.load('E:/pyramidal/jingCode/test.npy')
num_images=2
all_image=reconstruct(row,col,num_images,mat)
plt.imsave('E:/pyramidal/jingCode/1.png',all_image[1,:],format="png", cmap="gray")

