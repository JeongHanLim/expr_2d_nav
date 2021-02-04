import pickle
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

######################################
## Lidar : (length, 141->128)       ##
## Img   : (4, 144, 256, 3)         ##
######################################

def check_all_data():
    total_data = 0
    for episode in range(100):
        print("Processing on ", episode, "th episode...======================")
        suc, lists = Preprocess(episode).load_episode_data()
        if suc == True:
            total_data += len(lists)
            print(episode, "th episode has ", len(lists), " data. Total: ", total_data)
        else:
            pass
    return total_data


class Preprocess:
    def __init__(self, __episode):
        self.data_saved_path = 'D:/AirSimData/'
        self.episode = __episode

    # obs_lidar : [goaldata[3], position[3], velocity[3], orientation[4], lidar_data[16*8]]
    # obs_img : [images[0], images[1], ..., images[camera_type]], (4, 144, 256, 3)
    def load_episode_data(self):
        lists = []

        # Loading data from pickle.
        path = self.data_saved_path + "episode_" + str(self.episode)

        try:
            f = open(path, "rb")
        except:
            return False, []
        # There are some blank episodes from 0 ~ 100

        with f:
            lists = pickle.load(f)
            print("loading successful at " + path)
        return True, lists

    def extract_data_set(self, lists, start, size):
        lidar = np.empty((0, 141))
        img = np.empty((4, 144, 256, 3))
        obs_lidar = np.empty((0, 141))
        obs_img = np.empty((4, 144, 256, 3))
        sampled_list = np.empty(np.size(lists))

        if not start+size < len(lists):
            exit(0)

        for idx in range (start, start+size):
            sampled_list = lists[start:min(start+size, len(lists))]

#        for idx, traj in enumerate(lists):
        for _, traj in enumerate(sampled_list):
            obs_lidar, obs_img = traj["obs"]

            #self.check_img_valid(obs_img)
            lidar = np.vstack([lidar, obs_lidar])
            img = np.vstack([img, obs_img])

        print(lidar.shape)
        print(img.shape)
        return lidar, img


    def pure_lidar(__lidar_data):
        return __lidar_data[:, 13:]

    def check_img_valid(__image):
        for i in range(1, 254):
            if len(np.argwhere(__image == i)) > 0:
                print(i)
                plt.imshow(__image[0])
                plt.show()


    def len_run(lists):
        return len(lists)


if __name__ == "__main__":
#    episode = 0 #For Test

#    lists = Preprocess(episode).load_episode_data()
#    for idx in range(len(lists)):
#        lidar_data, img_data = Preprocess(episode).extract_data_set(lists, idx, 4)
        #print(idx," ", lidar_data.shape)

        #lidar_data = lidar_data.view(lidar_data.size(0), -1)
#        lidar_data = Variable(torch.tensor(lidar_data))
#        if torch.cuda.is_available():
#            lidar_data = lidar_data.cuda()

    print(check_all_data())

