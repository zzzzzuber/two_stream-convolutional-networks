import os
import pickle
import config
import gc
from collections import OrderedDict

def get_all_video_list():  # 将文件中的所有数据创建为视频名：视频类别
    video_list = []
    video_name_list = []
    video_file = {}
    video_list = os.listdir(config.dataset_file)

    for video in video_list:
        video_name_list = os.listdir(os.path.join(config.dataset_file, video))
        for video_name in video_name_list:
            video_name = str(video_name).split('.')[0]
            video_file[video_name] = video
    with open(config.data_file + 'video_list.pickle', 'wb') as fw:
        pickle.dump(video_file, fw)
        print("successfully dump all the video list")
        #format : 'v_CricketBowling_g05_c04': 'CricketBowling'
get_all_video_list()


def get_batch_video_list(batch_size=120):
    # the sum of videos = 120 batch_size x 111 batch
    # 13320 = 120 x 111
    with open(config.data_file + 'video_list.pickle', 'rb') as fr:
        video_list = pickle.load(fr)
    video_list = OrderedDict(video_list)
    video_sub_list = {}
    video_name = list(video_list.keys())
    # print(len(video_list))
    n_batch = len(video_list) // batch_size
    # print(n_batch)
    for batch in range(n_batch):
        video_sub_list = {}
        start_index = batch * batch_size
        print("the batch %d is starting!" % batch)
        # cut the number of batch_size videos to build some sub collections, i is the start position in video_list
        for i in range(batch_size):
            video_sub_list[video_name[start_index]] = video_list[video_name[start_index]]
            start_index = start_index + 1
            print(start_index)
        with open(config.data_file + 'video_sub_list_' + str(batch) + '.pickle', 'wb') as fsw:
            pickle.dump(video_sub_list, fsw)
            print("dump sub list %d successfully!" % batch)
    gc.collect()

get_batch_video_list(batch_size=120)
