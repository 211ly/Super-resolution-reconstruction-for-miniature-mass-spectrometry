import os
import sys
import json
import numpy as np
import math
import h5py
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from itertools import combinations
from tqdm import tqdm
import random

from scipy.signal import find_peaks


def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) / sigma) ** 2 / 2) / (np.sqrt(2 * np.pi) * sigma)


def generate_spectrum_combinations(ms_data, mass_range=(100, 700), num_points_per_mz=250, min_isotopes=None):
    """
    Generate simulated spectrum including all base isotopes and different numbers of variable isotopes.
    Min_isotopes means including how many variable isotopes at least.
    """

    num_points = num_points_per_mz*(mass_range[1]-mass_range[0])
    mass_values = np.linspace(mass_range[0], mass_range[1], num_points)
    spectrum = np.array([mass_values])
    base_isotopes, isotopes = list(ms_data['base_isotopes']), list(ms_data['isotopes'])
    resolution = ms_data['resolution']
    if min_isotopes is None:
        min_isotopes = len(isotopes)
    for r in tqdm(range(min_isotopes, len(isotopes)+1), desc="Generating Spectra"):
        for combination in combinations(isotopes, r):
            intensity_values = np.zeros_like(mass_values)
            for i in base_isotopes:
                for mass_str, intensity in i.items():
                    mass = np.float64(mass_str)
                    FWHM = mass / resolution
                    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
                    gau = gaussian(mass_values, mass, sigma)
                    gau /= max(gau)
                    intensity_values += intensity * gau

            for i in combination:
                for mass_str, intensity in i.items():
                    mass = np.float64(mass_str)
                    FWHM = mass / resolution
                    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
                    gau = gaussian(mass_values, mass, sigma)
                    gau /= max(gau)
                    intensity_values += intensity * gau
            spectrum = np.append(spectrum, [intensity_values], axis=0)
    return spectrum.T

def save_to_h5(dir_path, name, simulate_data, mode='w'):
    """Save simulated spectrum to a H5 file"""
    with h5py.File(dir_path, mode) as f:
        f.create_dataset(name, data=simulate_data)

def add_description_h5(file_name, description, dataset_name=None):
    """
    Add description attribute to the h5 file
    if dataset_name is None add description to the file's attrs
    """
    if dataset_name is None:
        with h5py.File(file_name, 'a') as F:
            F.attrs['description'] = description
    else:
        with h5py.File(file_name, 'a') as F:
            if dataset_name not in F:
                print(f"Dataset {dataset_name} is not in file.")
                print(F.keys())
                return
            else:
                dataset = F[dataset_name]
                dataset.attrs['description'] = description

def show_h5_file(file_path):
    x = np.linspace(100, 700, 150000)
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data = f[key][:]
            x = np.linspace(100, 700, 150000)
            plt.plot(x,data[:, 12])
            plt.title(key)
            plt.show()

def normalize_and_round_keys(json_data):
    keys_to_round = np.linspace(100, 700, 150000)
    
    for key in json_data:
        if isinstance(json_data[key], list):
            for item in json_data[key]:
                max_value = max(item.values())
                new_keys = {}
                for k, v in item.items():
                    # 归一化处理
                    normalized_value = v / max_value
                    # 找到最接近的键
                    closest_key = min(keys_to_round, key=lambda x: abs(x - float(k)))
                    # 更新字典中的键
                    new_keys[closest_key] = normalized_value
                # 更新原始字典中的键
                item.clear()
                item.update(new_keys)
    
    return json_data

def generate_fake_spectrum(fake_isotopes, resolution, ms_range, num_points_per_mz):
    mz_values = np.linspace(ms_range[0], ms_range[1], num_points_per_mz*(ms_range[1]-ms_range[0]))
    fake_spectrum = np.zeros_like(mz_values)
    max_value = max(max(sub_dict.values()) for sub_dict in fake_isotopes)
    fake_isotopes = [{key: value/max_value for key,value in sub_dict.items()} for sub_dict in fake_isotopes]
    for isotope in fake_isotopes:
        for mass_str, intensity in isotope.items():
            mass = np.float32(mass_str)
            FWHM = mass / resolution
            sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
            gau = gaussian(mz_values, mass, sigma)
            gau /= max(gau)
            fake_spectrum += intensity * gau
                
    if np.isnan(fake_spectrum).any():
        raise ValueError(f"Nan is in fake spectrum {fake_isotopes}.")
    fake_spectrum /= max(fake_spectrum)
    fake_spectrum = np.around(fake_spectrum, 4)
    return fake_spectrum

def generate_fake_section(json_data, half_len=6, num_points_per_mz=250):

    with open(json_data, 'r') as f:
        ms_data = json.load(f)

    resolution = ms_data['resolution']
    peaks = ms_data['peaks']
    res = np.linspace(0, 10, half_len*2*num_points_per_mz+1)

    for p in peaks:
        for mass, intensity in p.items():
            if intensity == 1.:
                mz = float(mass)
        mz_value = np.linspace(mz-half_len, mz+half_len, 2*half_len*num_points_per_mz)
        section = np.zeros_like(mz_value)


        for i, (mass, intensity) in enumerate(p.items()):
            mass = np.float32(mass)
            if i == 0:
                mass = mz_value[len(mz_value)//2]
            else:
                diff = abs(mz_value-mass)
                mass = mz_value[np.argmin(diff)]
            FWHM = mass / resolution
            sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
            gau = gaussian(mz_value, mass, sigma)
            section += gau * intensity

        section /= max(section)
        section = np.round(section, 4)
        section = np.append(section, int(mz))
        res = np.vstack((res, section))

    return res[1:, :]

def mix_two_isotopes(isotopes_1: dict, factor: float = 0.6 ,isotopes_2: dict = None, num_points_per_mz=250, half_len=6, resolution=10000):

    for m, i in isotopes_1.items():
        if i == 1.0:
             mz = float(m)

    mz_value = np.linspace(mz-half_len, mz+half_len, half_len*2*num_points_per_mz)
    section = np.zeros_like(mz_value)
    for i, (mass, intensity) in enumerate(isotopes_1.items()):
        mass = np.float32(mass)
        if i == 0:
            mass = mz_value[len(mz_value)//2]
        else:
            diff = abs(mz_value - mass)
            mass = mz_value[np.argmin(diff)]

        FWHM = mass/resolution
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        gau = gaussian(mz_value, mass, sigma)
        section += gau*intensity
    if isotopes_2 is not None:
        for i, (mass, intensity) in enumerate(isotopes_2.items()):
            mass = np.float32(mass)
            diff = abs(mz_value - mass)
            mass = mz_value[np.argmin(diff)]
            FWHM = mass / resolution
            sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
            gau = gaussian(mz_value, mass, sigma)
            section += gau * intensity * factor

    section /= max(section)
    section = np.round(section, 4)

    return section

def mix_two_arr(arr_1, arr_2, mass, factor, num_points_per_mz=250):
    m1, m2 = mass
    move_points = int(abs(m1-m2) * num_points_per_mz)

    arr_add = np.zeros_like(arr_1)

    if m1 < m2:
        arr_add[move_points:] = arr_2[:-move_points]
        arr_add[:move_points] = arr_2[move_points:]
    else:
        arr_add[:-move_points] = arr_2[move_points:]
        arr_add[-move_points:] = arr_2[-move_points:]
    arr_add *= factor
    plt.plot(arr_add)
    plt.show()
    res = arr_add + arr_1
    return np.round(res/max(res), 4)

def generate_fake_dataset(json_data, num_of_spectrum, num_of_peaks, save_path, seed=12, mass_range=(100, 700), num_points_per_mz=250):
    
    spectrum_len = (mass_range[1]-mass_range[0])*num_points_per_mz
    mz_value = np.linspace(mass_range[0], mass_range[1], spectrum_len)
    fake_spectrum_dataset = np.array([mz_value])
    
    with open(json_data, 'r') as f:
        ms_data = json.load(f)
    h5_file_path = os.path.join(save_path, 'fake_spectrum_dataset.h5')

    all_isotopes = []
    for key in ms_data.keys():
        if isinstance(ms_data[key], list):
            all_isotopes += ms_data[key]
    resolution = ms_data["resolution"]

    st, ed = num_of_peaks
    
    for i in tqdm(range(num_of_spectrum), desc="Generating Fake Spectrum DataSet\n"):
        fake_isotopes = []
        
        np.random.seed(seed=seed+i) #保证每次生成的谱图都不同
        random.seed(seed+i)

        num_peaks = np.random.randint(st, ed) #本次生成的谱图中包含同位素峰组的个数
        print(f"Iteration {i} has {num_peaks} isotopes block.")
        factors = abs(np.random.normal(500, 500, num_peaks))
        selected_isotopes = random.sample(all_isotopes, num_peaks)

        for index, s in enumerate(selected_isotopes):
            for mz_str, intensity in s.items():
                if s[mz_str] * factors[index] < sys.float_info.max:
                    s[mz_str] *= factors[index]
            fake_isotopes.append(s)

        fake_spectrum = generate_fake_spectrum(fake_isotopes, resolution, mass_range, num_points_per_mz)
        fake_spectrum_dataset = np.append(fake_spectrum_dataset, [fake_spectrum], axis=0)

        
    fake_spectrum_dataset = fake_spectrum_dataset.T
    fake_spectrum_dataset = fake_spectrum_dataset[:, 1:]
    
    with h5py.File(h5_file_path, 'w-') as hf:
        hf.create_dataset('dataset', data=fake_spectrum_dataset)
    

                
