import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

from sklearn.decomposition import TruncatedSVD
import h5py



def alignment(ms, bin_size=250, min_mz=100, max_mz=700):
    mz, intensity = ms[:, 0], ms[:, 1]

    # Ensure that mz contains at least two elements within the desired range
    if len(np.where((mz >= min_mz) & (mz <= max_mz))[0]) < 2:
        raise ValueError("mz must contain at least 2 elements within the specified range")

    start, end = np.where(mz >= min_mz)[0][0], np.where(mz <= max_mz)[0][-1]

    len_of_spectrum = bin_size * (max_mz - min_mz)
    index, mzz = start, mz[start]
    res_mz, res_intensity = np.array([]), np.array([])

    while index <= end:
        if res_intensity.shape[0] == (max_mz-min_mz)*bin_size:
            break
        now_mz, now_intensity = np.array([]), np.array([])
        while (index <= end and mz[index] < mzz + 1):
            if now_mz.size == 0 or mz[index] > now_mz[-1]:
                now_mz = np.append(now_mz, mz[index])
                now_intensity = np.append(now_intensity, intensity[index])
            index += 1

        if now_mz.size < 2:
            print("Insufficient data points for cubic spline interpolation at mz range:", mzz, "-", mzz + 1)
            break

        cs = CubicSpline(now_mz, now_intensity)
        x_interp = np.linspace(mzz, mzz + 1, bin_size)
        y_interp = cs(x_interp)
        y_interp[y_interp < 0] = 0
        mzz += 1
        res_mz = np.concatenate((res_mz, x_interp))
        res_intensity = np.concatenate((res_intensity, y_interp))

    if res_intensity.shape[0] < len_of_spectrum:
        print("Spectrum fills with zero.")
        len_of_zero = len_of_spectrum - res_intensity.shape[0]
        fill = np.zeros(len_of_zero)
        res_intensity = np.concatenate((fill, res_intensity))
        res_mz = np.concatenate((fill, res_mz))

    print("Resulting intensity shape:", res_intensity.shape)

    return res_mz, res_intensity


def preprocess(dir_path, bin_size, min_mz, max_mz, save_path, key, mode):
    ms_size = (max_mz - min_mz) * bin_size
    mz = np.linspace(min_mz, max_mz, ms_size)
    res = np.zeros_like(mz)

    try:
        file_list = os.listdir(dir_path)
        file_list = [os.path.join(dir_path, f).replace('\\', '/') for f in file_list]
        num_of_files = len(file_list)

        for f in file_list:
            try:
                ms = np.array(pd.read_csv(f, header=None), dtype=np.float32)
                print(f"Processing file: {f}, Data shape: {ms.shape}")

                _, intensity = alignment(ms, bin_size, min_mz, max_mz)
                res = np.vstack((res, intensity))
            except Exception as e:
                print(f"Error in read files {f}: {e}")

    except Exception as e:
        print(f"Error in preprocess: {e}")

    res = res[1:, :]
    return res

    # with h5py.File(save_path, mode) as f:
    #     f.create_dataset(key, data=res)



def convert2signal(dir_path, align=True, bin_size=25, min_mz=100, max_mz=700):
    size_mz = (max_mz-min_mz)*bin_size
    try:
        file_list = os.listdir(dir_path)
        file_list = [os.path.join(dir_path, f).replace('\\', '/') for f in file_list]
        all_intensity = np.zeros((0, size_mz))  # assuming each spectrum is resized to 15000
        for f in file_list:
            try:
                ms = pd.read_csv(f, header=None).to_numpy(dtype=np.float64)
                if align:
                    mz, intensity = alignment(ms, bin_size, min_mz, max_mz)
                else:
                    mz, intensity = ms[:, 0], ms[:, 1]
                intensity[intensity < 0] = 0
                intensity /= np.max(intensity, initial=1)  # avoid division by zero
                all_intensity = np.vstack((all_intensity, intensity))
            except Exception as e:
                print(f"Error processing file {f}: {e}")
        signal = np.array([np.fft.irfft(i) for i in all_intensity])
        return signal
    except Exception as e:
        print(f"Error in convert2signal: {e}")
        return None

def svd_reconstruct(X, Y, N):
    svd = TruncatedSVD(n_components=N)
    svd.fit(X)
    subspace_basis = svd.components_
    Y_mapped = svd.transform(Y)
    Y_reconstructed = np.dot(Y_mapped, subspace_basis)
    return Y_reconstructed

def convert_spectrum_to_signal(spectrum_path, signal_path):
    """Convert a H5 file which includes spectrum to a H5 file with corresponding signals."""
    _, filename = os.path.split(spectrum_path)
    with h5py.File(spectrum_path, 'r') as spectrum_f:
        with h5py.File(signal_path, 'w') as signal_f:
            for key in spectrum_f.keys():
                data = spectrum_f[key][:]
                signal_data = np.zeros((data.shape[0]*2-2, data.shape[1]))
                print(signal_data.shape)

                for i in range(data.shape[1]):
                    signal_data[:, i] = np.fft.irfft(data[:, i], axis=0)

                signal_f[key] = signal_data



