import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import h5py
import pandas as pd
from scipy.stats import pearsonr



class MSDataSet(Dataset):
    def __init__(
            self,
            file_path,
            min_mz,
            max_mz,
            num_points_per_mz,
            ):
        super().__init__()
        self.file_path = file_path
        self.data_dir, self.file_name = os.path.split(self.file_path)
        _, self.file_type = os.path.splitext(self.file_name)
        self.mass_range = (min_mz, max_mz)
        self.spectrum_len = (self.mass_range[1] - self.mass_range[0]) * num_points_per_mz


        if self.file_type == '.h5':

            with h5py.File(self.file_path, 'r') as f:
                self.len = 0
                self.data = []
                for key in f.keys():
                    if key != 'description':
                        spectrum_data = np.array(f[key][:].T, dtype=np.float32)

                        if spectrum_data.shape[1] != self.spectrum_len:
                            raise ValueError(f"The len of {key} is not equal true len.")

                        if len(self.data) == 0:
                            self.data = spectrum_data
                        else:
                            self.data = np.concatenate([self.data, spectrum_data], axis=0)

                        self.len += spectrum_data.shape[0]
        else:
            raise ValueError("Can't deal with this file type.")


    def __getitem__(self, index):
        return self.data[index,:]

    def __len__(self):
        return self.len




class BrickMSDataSet(Dataset):
    def __init__(
            self,
            brick_path,
            simulate_path,
            min_mz,
            max_mz,
            num_points_per_mz,
            ):
        super().__init__()
        self.brick_path, self.simulate_path = brick_path, simulate_path
        self.data_dir, self.file_name = os.path.split(self.brick_path)
        _, self.file_type = os.path.splitext(self.file_name)
        self.mass_range = (min_mz, max_mz)
        self.spectrum_len = (self.mass_range[1] - self.mass_range[0]) * num_points_per_mz


        if self.file_type == '.h5':

            with h5py.File(self.brick_path, 'r') as f:
                with h5py.File(self.simulate_path, 'r') as s_f:

                    self.len = 0
                    self.data, self.target = np.array([]), np.array([])

                    for key in f.keys():
                        if key != 'description':
                            spectrum_data = np.array(f[key][:].T, dtype=np.float32)
                            spectrum_data = np.array([s/max(s) for s in spectrum_data])
                            if spectrum_data.shape[1] != self.spectrum_len:

                                raise ValueError(f"The len of {key} is not equal true len.")

                            if len(self.data) == 0:
                                self.data = spectrum_data
                            else:
                                self.data = np.concatenate([self.data, spectrum_data], axis=0)

                            simulate_data = np.array(s_f[key][:].T, dtype=np.float32)
                            # similarity = cosine_similarity(spectrum_data, simulate_data)
                            distance = euclidean_distances(spectrum_data, simulate_data)
                            for d in distance:
                                most_similar = simulate_data[np.argmin(d)].reshape(1, -1)
                                if len(self.target) == 0:
                                    self.target = most_similar
                                else:
                                    self.target = np.concatenate([self.target, most_similar], axis=0)



                            self.len += spectrum_data.shape[0]
        else:
            raise ValueError("Can't deal with this file type.")


    def __getitem__(self, index):
        return self.data[index, :], self.target[index, :]

    def __len__(self):
        return self.len

class SectionDataSet(Dataset):
    def __init__(self,
                 brick_path,
                 simulate_path,
                 half_len,
                 num_points_per_mz):
        super().__init__()
        self.brick_path = brick_path
        self.brick_data = np.array(pd.read_csv(brick_path, header=None))
        self.simulate_data = np.array(pd.read_csv(simulate_path, header=None))
        self.len_of_section = half_len * num_points_per_mz * 2
        self.mz_key, self.target_key = set(self.brick_data[:, -1]), set(self.simulate_data[:, -1])
        self.target_index = {t: i for i, t in enumerate(self.simulate_data[:, -1])}
        self.brick_index = {i: t for i, t in enumerate(self.brick_data[:, -1])}


    def __getitem__(self, i):
        brick_data, mz = self.brick_data[i, :-1], self.brick_data[i, -1]
        simulate_data, key_mz = self.simulate_data[self.target_index[mz], :-1], self.simulate_data[self.target_index[mz], -1]
        if mz != key_mz:
            raise ValueError("Not match.")

        return brick_data, simulate_data

    def __len__(self):
        return self.brick_data.shape[0]

    def get_mz(self, i):
        return self.brick_index[i]

    def add_noise(self, num, factor=0.001, base_seed=42):
        res = np.zeros((1, self.len_of_section+1))
        for j in range(len(self)):
            data, mz = self.brick_data[j, :-1], self.brick_data[j, -1]
            for i in range(num):
                np.random.seed(base_seed+i)
                noise = np.random.rand(self.len_of_section) * factor
                data += noise
                data /= max(data)
                data = np.round(data, 4)
                data_with_mz = np.append(data, mz)
                res = np.vstack([res, data_with_mz])
        path = self.noise_path()
        df = pd.DataFrame(res[1:, :])
        df.to_csv(path, header=False, index=False, mode='w')

    def noise_path(self):
        directory, filename = os.path.split(self.brick_path)
        file, file_type = os.path.splitext(filename)
        file += '_with_noise'
        new_path = os.path.join(directory, file+file_type).replace('\\', '/')
        return new_path




def MS_dataset_collate(batch):
    res = []
    for x in batch:
        res.append(x)
    data_raw, data_target = torch.from_numpy(np.array(res, np.float32)), torch.from_numpy(np.array(res, np.float32))
    return data_raw, data_target


def Brick_dataset_collate(batch):
    sample, target = [], []
    for x in batch:
        sample.append(x[0])
        target.append(x[1])
    data_raw, data_target = torch.from_numpy(np.array(sample,dtype=np.float32)),torch.from_numpy(np.array(target,dtype=np.float32))
    return data_raw, data_target

def Section_dataset_collate(batch):
    sample, target = [], []
    for x in batch:
        sample.append(x[0])
        target.append(x[1])
    data_raw, data_target = torch.from_numpy(np.array(sample, dtype=np.float32)), torch.from_numpy(
        np.array(target, dtype=np.float32))
    return data_raw, data_target




# if __name__ == '__main__':
#     data = SectionDataSet('../data/Brick/preprocess/section.csv',
#                              '../simulate-data/section.csv',
#                              6, 250)
#     data.add_noise(10)