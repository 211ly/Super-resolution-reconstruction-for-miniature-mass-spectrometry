import os
import time

import pandas as pd
from spec2signal import alignment
from scipy.signal import find_peaks
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from datetime import datetime
import h5py

from nets.LSTM import Seq2SeqLSTM

class Reconstruct:
    _defaults = {
        'model_path': './model_data/LSTM/Epoch200-910.pth',
        'brick_path': './data/Brick/preprocess/section.csv',
        'cuda': True,
        'input_size': 250,
        'hidden_size': 128,
        'output_size': 250,
        'seq_len': 12,
        'split_size': 250
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)

        for key, value in kwargs.items():
            self.__setattr__(key, value)

        self.net = self.generate_trained_net()

    def generate_trained_net(self):

        net = Seq2SeqLSTM(self.input_size, self.hidden_size, self.output_size,
                          self.seq_len, self.split_size)

        device = 'cuda' if self.cuda else 'cpu'
        net.load_state_dict(torch.load(self.model_path, map_location=device))
        net.eval()

        if self.cuda:
            net = net.cuda()

        return net

    def rec(self):
        path = self.brick_path
        data = np.array(pd.read_csv(path, header=None))
        section, mz = data[:, :-1], data[:, -1]
        section = torch.tensor(section, dtype=torch.float32).cuda()
        section = self.net(section)
        section = section.squeeze(1)
        section = section.cpu().detach().numpy()
        section = np.array([np.round(abs(s)/max(s), 4) for s in section])

        return section, mz

    def rec_section(self, intensity):
        section = torch.tensor(intensity, dtype=torch.float32).cuda()
        section = self.net(section)
        section = section.squeeze(1)
        section = section.cpu().detach().numpy()
        section = np.array([np.round(abs(s) / max(s), 4) for s in section])
        return section

    def rec_spectrum(self, dir_path, save=False):
        path = os.listdir(dir_path)
        read_path = [os.path.join(dir_path, p).replace('\\', '/') for p in path]
        file_names = [p.split('.')[0] for p in path]

        for i, pt in enumerate(read_path):
            raw_data = np.array(pd.read_csv(pt, header=None, encoding='UTF-8', sep=','))
            raw_mz, raw_intensity = alignment(raw_data, min_mz=175, max_mz=850)
            raw_intensity /= max(raw_intensity)

            output_intensity = np.zeros_like(raw_intensity)

            peaks, _ = find_peaks(raw_intensity, distance=8*250, height=0.05*max(raw_intensity))
            target = np.array([raw_intensity[p-1500:p+1500] for p in peaks if p+1500<raw_intensity.shape[0]])
            target = np.array([np.round(t/max(t), 4) for t in target])

            target = self.rec_section(target)

            for p, t in zip(peaks, target):
                output_intensity[p-1500:p+1500] = t*raw_intensity[p]
                peak, _ = find_peaks(t, height=0.001*max(t), distance=250)
                print(t[peak], raw_mz[p])
                for pp in peak:
                    print(np.round((pp-3*250)/250+raw_mz[p], 4))


            output_intensity /= max(output_intensity)

            if save:
                file_name = file_names[i] + '-rec.csv'
                df = pd.DataFrame(np.array([raw_mz, output_intensity]).T)
                df.to_csv(os.path.join(dir_path,file_name), index=False, header=False,mode='w')



if __name__ == '__main__':


    r = Reconstruct()



    r.rec_spectrum('./data/Brick/raw/bac', save=True)

    dir = './data/Brick/raw/bac'
    path_list = os.listdir(dir)
    path_list = [os.path.join(dir, p) for p in path_list if p.endswith('-rec.csv')]

    for p in path_list:
        data = np.array(pd.read_csv(p))
        mz, intensity = data[:, 0], data[:, 1]
        plt.plot(mz, intensity)
        plt.title(p)
        plt.show()










