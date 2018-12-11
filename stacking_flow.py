
import numpy as np
import os
import cv2

'''
叠加光流的步骤：
1.读取光流图像
2.对其进行裁切，将其转换为（224，224）大小的图像
3.append进光流的列表中，x和y方向交替进行
4.np.array()将列表矩阵化。
'''
def stack_flow(path):
    optical_flow_stack = []
    #path = os.path.join(root_file,foldername)+'/'
    flow_images = os.listdir(path)
    print("stack_flow path:%s"%path)
    for i in range(len(flow_images)):
        image = cv2.imread(os.path.join(path,flow_images[i]))
        #！未对图像进行随机裁剪，目前先测试用resize对图像大小进行变换。
        image = cv2.resize(image,(224,224))
        #print("image size : %s"%str(image.shape()))
        img = image-np.mean(image)
        img = img/255.
        optical_flow_stack.append(img)
    #cv2.imread return an image which channels are BGR,this part would be checked
    optical_flow_stack = np.array(optical_flow_stack)
    #将图像一张一张叠起来，放到一个列表里，再对其进行矩阵化，最后维度为：[frame_numbers,[224,224]]
    #print(type(optical_flow_stack))
    #optical_flow_stack = np.swapaxes(optical_flow_stack, 0, 1)
    #optical_flow_stack = np.swapaxes(optical_flow_stack, 1, 2)
    #optical_flow_stack saves one video all the optical flow images
    #print("the shape of optical_flow_stack:%s"%str(optical_flow_stack.shape()))
    return optical_flow_stack