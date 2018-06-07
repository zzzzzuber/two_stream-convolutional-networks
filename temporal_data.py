"""
    视频名_帧索引
    按视频类别分文件夹
    按方向分隔文件夹
"""
import cv2
import os
import numpy as np
import tensorflow as tf
import pickle

root_file = 'E:/tensorflow_workspace/two_stream_zhuc/'
dataset_file = root_file + 'UCF-101/'
split_file = root_file + 'ucfTrainTestlist/'
flow_file = root_file + 'flow/'
data_file = root_file + 'data/'


def optical_flow(file_path, save_path):
    frame_dict = {}
    cap = cv2.VideoCapture(file_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame_count = 1
    while (1):
        ret1, frame2 = cap.read()
        if ret1:
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # CALCULATE FLOW
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # x方向：flow[...,0]
            # y方向：flow[...,1]
            frame_dict['frame_id'] = 'frame_optical_'+str(frame_count)
            frame_dict['x'] = flow[...,0]
            frame_dict['y'] = flow[...,1]

            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2', rgb)
            cv2.waitKey(30)
            cv2.imwrite(save_path + '/' + 'frame' + str(frame_count) + '.jpg', rgb)
            frame_count = frame_count + 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame_dict

def get_all_data():
    split_file_path = os.path.join(split_file, 'trainlist01.txt')
    video_dict = {}
    '''
        视频字典格式：
        video_dict{
            video_name:string type
            class_name:string type
            frame_dict:{
                frame_id :int
                x:float
                y:float
            }
        }
    '''
    with open(split_file_path, 'r') as f:
        lines = f.readlines()
        frame_dict = {}
        for line in lines:
            line = str(line)  # 逐行读取文件
            video_name = line.split()[0]  # 从txt文件中切割出视频名（带路径）如：ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi
            class_name = line.split('/')[0]  # 从txt文件中切割出类别名
            v = video_name.split('.')[0]  # 不带avi后缀名的视频名
            video_middle_name = video_name.split('/')[1].split('.')[0]  # 如：v_ApplyEyeMakeup_g08_c01
            # 对视频路径操作
            video_path = os.path.join(dataset_file, video_name)  # 根据视频路径查找视频
            save_path_c = os.path.join(flow_file, class_name) + '/'  # 逐级创建目录
            save_path_v = os.path.join(flow_file, v) + '/'
            if not os.path.exists(save_path_c):
                os.mkdir(save_path_c)
            if not os.path.exists(save_path_v):
                os.mkdir(save_path_v)

            frame_dict = optical_flow(video_path, save_path_v)

            # 对视频字典操作
            video_dict[video_name] = video_middle_name
            video_dict[class_name] = class_name
            video_dict['frame_dict'+video_name] = {}
            video_dict['frame_dict'+video_name] = frame_dict

            #print("==============")
            #print(video_dict)
    return video_dict

def create_pickle(video_dict):
    with open(root_file+'data/'+'video_dict.pickle','wb') as f:
        pickle.dump(video_dict,f)
        print("successfully dump data")
video = {}
video = get_all_data()
create_pickle(video)
'''
with open(root_file+'data/'+'video_dict.pickle','rb') as fr:
    k = pickle.load(fr)
    print(k)
'''
def get_all_videoname():
    video_list = {}
    classes = os.listdir(dataset_file)
    for cnames in classes:
        videos = os.listdir(os.path.join(dataset_file,cnames))
        for video in videos:
            video_list[video] = cnames

    with open(data_file+'video_list.pickle','wb')as fw:
        pickle.dump(video_list,fw)
        print("successfully！")
