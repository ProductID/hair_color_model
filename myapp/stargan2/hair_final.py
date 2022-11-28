#!/usr/bin/env python
# coding: utf-8

# In[65]:


import cv2
import os
import numpy as np
from skimage.filters import gaussian
# from test import evaluate
from .test_h import evaluate
# import matplotlib.pyplot as plt
# from PIL import Image
# # img = cv2.cvtColor(orip, cv2.COLOR_BGR2RGB)
# # #
# # display(Image.fromarray(img))
#
# from sklearn import neighbors, datasets
#
# # In[66]:
#
#
# import glob
# from matplotlib import pyplot as plt
import cv2
# all_images=glob.glob('./imgs/*')


# In[67]:


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


# In[85]:

def main_hair(image_path,color_code,save_path):
    # image_path='./imgs/5.jpg'
    b, g, r = color_code  #[10, 50, 250]       # [10, 250, 10]
    print('Model loding.......................')

    cp = 'myapp/stargan2/cp/79999_iter.pth'
    cp = '/var/www/html/celebrity_model-master/myapp/stargan2/cp/79999_iter.pth'

    image = cv2.imread(image_path)
    print('image read done.................')
    row,col=image.shape[0:2]

    ori = image.copy()
    parsing = evaluate(image_path, cp)
    print('mode evalution done....................................')
    parsing = cv2.resize(parsing, (col,row), interpolation=cv2.INTER_NEAREST)

    changed_M = np.zeros_like(image)
    ori = image.copy()
    # masked_lips = cv2.bitwise_and(im, im, mask=mask[:, :, 0])

    changed_M[parsing == 17] = ori[parsing == 17]
    # plt.imshow(changed_M[:,:,::-1])
    # plt.show()


    # In[84]:


    row,col=image.shape[0:2]
    # row


    # In[86]:


    Dst=changed_M
    # plt.imshow(Dst[:,:,::-1])
    # plt.show()


    # In[87]:


    changed_=changed_M
    del changed_M
    # tar_color = np.zeros_like(image)

    changed_[:,:,0]=np.where(changed_[:,:,0]==0,changed_[:,:,0], b)
    changed_[:,:,1]=np.where(changed_[:,:,1]==0,changed_[:,:,1], g)
    changed_[:,:,2]=np.where(changed_[:,:,2]==0,changed_[:,:,2], r)
    new_imageb=((np.where(parsing==0,parsing, 255))).astype('uint8')

    # new_imageb = ((np.where(parsing == 17, parsing, 0) / 17) * 255).astype('uint8')

    tar_color = np.zeros_like(image)
    tar_color[:,:,0]=new_imageb
    tar_color[:,:,1]=new_imageb
    tar_color[:,:,2]=new_imageb
    # plt.imshow(changed_[:,:,::-1])
    # plt.show()


    # In[91]:
    print('image reading again................117')
    image = cv2.imread(image_path)
    row,col=image.shape[0:2]

    ori = image.copy()
    parsing = evaluate(image_path, cp)
    print('model evalution again done.................................')
    parsing = cv2.resize(parsing, (col,row), interpolation=cv2.INTER_NEAREST)

    changed_M = np.zeros_like(image)
    ori = image.copy()
    # masked_lips = cv2.bitwise_and(im, im, mask=mask[:, :, 0])

    changed_M[parsing == 17] = ori[parsing == 17]
    # plt.imshow(changed_M[:,:,::-1])
    # plt.show()


    # In[84]:


    row,col=image.shape[0:2]
    # row


    # In[86]:


    Dst=changed_M

    # image_1=cv2.imread('/home/aj/Documents/face-makeup./changed_image.png')
    # image_d=cv2.imread('/home/aj/Documents/face-makeup./Dst_image.png')
    # image_m=cv2.imread('/home/aj/Documents/face-makeup./tar_color_image.png')
    image_1=changed_
    image_d=Dst
    image_m=tar_color
    # print('I1')
    # plt.imshow(image_1[:,:,::-1])
    # plt.show()
    # print('I2')
    # plt.imshow(image_d[:,:,::-1])
    # plt.show()
    # print('I3')
    # plt.imshow(image_m[:,:,::-1])
    #
    # plt.show()


    # In[60]:


    # # This is where the CENTER of the airplane will be placed
    center = (int(col/2),int(row/2))


    monoMaskImage = cv2.split(image_m)[0] # reducing the mask to a monochrome
    br = cv2.boundingRect(monoMaskImage) # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)

    # # Clone seamlessly.
    des=image_d
    # plt.imshow(des[:, :, ::-1])
    # plt.show()
    # output_ = cv2.seamlessClone(image_1, image_d, image_m[:,:,0], centerOfBR, cv2.MIXED_CLONE)


    # plt.imshow(output_[:, :, ::-1])
    # plt.show()
    #
    image_d_D= np.zeros_like(image)
    image_d_D[:,:,0]=(image_d[:,:,0]/255)*b
    image_d_D[:,:,1]=(image_d[:,:,1]/255)*g
    image_d_D[:,:,2]=(image_d[:,:,2]/255)*r
    output_ = cv2.seamlessClone(des, image_d_D, image_m[:,:,0], centerOfBR, cv2.MIXED_CLONE)
    # plt.imshow(output_[:, :, ::-1])
    # plt.show()

    # output_=image_d

    # # Save result

    # # cv2.imwrite("images/opencv-seamless-cloning-example.jpg", output);


    # In[61]:


    # plt.imshow(output_[:,:,::-1])
    # plt.show()


    # In[62]:


    # plt.imshow(image_d[:,:,::-1])
    # plt.show()
    #

    # In[64]:


    col_image=output_
    # col_image=sharpen(image_d)

    orip = image.copy()


    # orip[np.logical_or(parsing == 13 ,parsing == 12)] = col_image[np.logical_or(parsing == 13 ,parsing == 12)]
    col_image[parsing != 17] = orip[parsing != 17]

    # plt.imshow(col_image[:,:,::-1])
    # plt.show()


    # img = cv2.cvtColor(col_image, cv2.COLOR_BGR2RGB)

    # final_imag = np.hstack((orip, col_image))
    final_imag =col_image


    final_image = cv2.cvtColor(final_imag,cv2.COLOR_BGR2RGB)

    cv2.imwrite(save_path, final_imag)
    print(save_path,'ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
    # final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

    return final_image
    #
    # plt.imshow(Image.fromarray(final_image))
    #plt.show()

# In[ ]:

# Name: Falu Red
# RGB: (24, 24, 128)

# Name: Brown Chocolate
# RGB: (52, 25, 99)

# Name: Pansy Purple
# RGB: (77, 34, 121)

# Name: Palatinate Purple
# RGB: (96, 40, 104)



color_code=[77, 34, 121]
# count =1
# for i in glob.glob('imgs/*'):
#     main_hair(i,color_code,count)
#     count=count+1

