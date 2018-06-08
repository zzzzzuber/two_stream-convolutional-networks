
import cv2
import os
import numpy as np
import tensorflow as tf
import pickle

root_file = 'E:/tensorflow_workspace/two_stream_zhuc/'
dataset_file = root_file + 'UCF101/'
split_file = root_file + 'ucfTrainTestlist/'
flow_file = root_file + 'flow/'
data_file = root_file + 'data/'

video_dict = {}

def get_all_video_list(): #将文件中的所有数据创建为视频名：视频类别
    video_list = []
    video_name_list = []
    video_file = {}
    video_list = os.listdir(dataset_file)
    
    for video in video_list:
        video_name_list = os.listdir(os.path.join(dataset_file,video))
        for video_name in video_name_list:
            video_name = str(video_name).split('.')[0]
            video_file[video_name]= video
    with open(data_file+'video_list.pickle','wb') as fw:
        pickle.dump(video_file,fw)
        print("successfully dump all the video list")
get_all_video_list()
#with open(data_file+'video_list.pickle','rb') as fr:
#    print(pickle.load(fr))

def optical_flow(file_path,save_path,video_name):
    frame_dict = {}
    cap = cv2.VideoCapture(file_path)
    ret,frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    frame_count = 1
    while (1):
        ret1,frame2 = cap.read()
        if ret1:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            #CALCULATE FLOW
            flow = cv2.calcOpticalFlowFarneback(prvs,next,None,0.5,3,15,3,5,1.2,0)
            mag,ang = cv2.cartToPolar(flow[...,0],flow[...,1])
            #x方向：flow[...,0]
            #y方向：flow[...,1]
            frame_dict['frame_'+str(frame_count)] = [flow[...,0],flow[...,1]]
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow(video_name,rgb)
            cv2.waitKey(30)
            cv2.imwrite(save_path+str(frame_count)+'.jpg',rgb)
            prvs = next 
            frame_count = frame_count+1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame_dict
'''
print(optical_flow('E:/tensorflow_workspace/two_stream_zhuc/UCF101/ApplyLipstick/v_ApplyLipstick_g01_c01.avi',
             'E:\tensorflow_workspace\two_stream_zhuc\flow\ApplyLipstick\v_ApplyLipstick_g01_c01',
             'v_ApplyLipstick_g01_c01'
             )
    )
'''

def get_all_optical_flow():
    video_list = []
    frame_dict = {}
    with open(root_file+'data/'+'video_list.pickle','rb') as fr:
        video_list = pickle.load(fr)
        print("successfully load all video_list data")
        #print(video_list)
    
    for video in video_list:
        
        #create save file path
        save_path_c = flow_file +video_list[video]+'/'
        if not os.path.exists(save_path_c):
            os.mkdir(save_path_c) #create class name file folders
        save_path_v = save_path_c + video + '/'
        if not os.path.exists(save_path_v):
            os.mkdir(save_path_v) #create video name file folders
            
        video_path = dataset_file + video_list[video] +'/'+ video +'.avi'
        print(video_path)
        #calculate the optical flow and create save dict
        frame_dict = optical_flow(video_path,save_path_v,video)
        
        #save all the frames information
        video_dict[video] = {}
        video_dict[video]['class_name'] = video_list[video]
        video_dict[video]['frame_dict'] = frame_dict
    with open(data_file+'video_dict.pickle','wb') as fv:
        pickle.dump(video_dict,fv)
        print("successfully dump all the frame information!")
get_all_optical_flow()    
with open(data_file+'video_dict.pickle','rb') as fk:
    print(pickle.load(fk))
