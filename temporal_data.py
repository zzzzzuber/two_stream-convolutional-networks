import cv2
import os
import numpy as np
import pickle
import gc
import tqdm
import config
from collections import OrderedDict
root_file = config.root_file
dataset_file = config.dataset_file
split_file = config.split_file
flow_file = config.flow_file
data_file = config.data_file

video_dict = {}
def pixel_norm(flow):
    min = np.min(flow)
    max = np.max(flow)
    value = (flow-min)/(max-min)
    value = value *255.0
    return value
def optical_flow(file_path, save_path, video_name):
    frame_dict = {}
    cap = cv2.VideoCapture(file_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    frame_count = 1
    while (1):
        ret1, frame2 = cap.read()
        if ret1:
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # CALCULATE FLOW
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # x方向：flow[...,0]
            # y方向：flow[...,1]
            x_flow = flow[...,0]
            y_flow = flow[...,1]
            x = np.array(pixel_norm(x_flow),dtype=np.float64)
            #print(np.shape(x))   x->(240,320)
            y = np.array(pixel_norm(y_flow),dtype=np.float64)

            cv2.waitKey(30)
            cv2.imshow(video_name, x)
            cv2.imshow(video_name+'2',y)
            cv2.imwrite(save_path + 'frame_' + str(frame_count) + '_x'+'.jpg', x)
            cv2.imwrite(save_path + 'frame_' + str(frame_count) + '_y'+'.jpg', y)
            prvs = next  ###########################
            frame_count = frame_count + 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    gc.collect()
#optical_flow('E:/tensorflow_workspace/two_stream_zhuc/UCF101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi', 'E:/tensorflow_workspace/two_stream_zhuc/test/', 'v_ApplyEyeMakeup_g01_c01')

def get_all_optical_flow():
    video_list = {}
    frame_dict = {}
    batch_size = 120
    n_batch = 111
    with open(root_file + 'data/' + 'video_list.pickle', 'rb') as fr:
        video_list = pickle.load(fr)
        print("successfully load all video_list data")
        # print(video_list)
    for batch in range(n_batch):
        with open(data_file+'video_sub_list_'+str(batch)+'.pickle','rb') as frb:
            video_sub_list = pickle.load(frb)
        print('###################################')
        print("the %d batch starting running!" % batch)
        for video in tqdm.tqdm(video_sub_list):
            #Error: Insufficient memory (Failed to allocate 1536000 bytes) in cv::OutOfMemoryError,
            # create save file path
            save_path_c = flow_file + video_list[video] + '/'
            if not os.path.exists(save_path_c):
                os.mkdir(save_path_c)  # create class name file folders
            save_path_v = save_path_c + video + '/'
            if not os.path.exists(save_path_v):
                os.mkdir(save_path_v)  # create video name file folders

            video_path = dataset_file + video_list[video] + '/' + video + '.avi'
            print(video_path)
            # calculate the optical flow and create save dict
            try:
                optical_flow(video_path, save_path_v, video)
            except Exception as e:
                gc.collect()

        gc.collect()
    gc.collect()
get_all_optical_flow()
