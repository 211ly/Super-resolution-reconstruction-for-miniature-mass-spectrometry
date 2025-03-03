# A Bi-LSTM-based super-resolution reconstruction algorithm for miniature mass spectrometry

**In mass spectrometry, achieving high resolution often involves a**
**trade-off with the signal-to-noise ratio (SNR), particularly in miniature mass**
**spectrometers, where both properties typically fall short compared to**
**commercial instruments. This project introduces a novel super-resolution**
**reconstruction algorithm based on Bidirectional Long Short-Term Memory**
**(Bi-LSTM) networks. This algorithm effectively overcomes hardware performance**
**limitations and reconciles the trade-off between SNR and resolution. The**
**trained Bi-LSTM can perform super-resolution reconstruction of fixed-length**
**mass spectral segments.**

![1740969352543](C:\Users\YI\AppData\Roaming\Typora\typora-user-images\1740969352543.png)

# Installation

Python and Pytorch:

Python 3.10.8 and Pytorch 2.1.0+cu118

# Clone the repo and run it directly

```
git clone https://github.com/211ly/Super-resolution-reconstruction-for-miniature-mass-spectrometry
```

# Download the model and run it

## **Training the neural network**

run the file train_lstm.py to train the neural network

the utils contain the loading of data and record the performance of the model training and the nets contain the neural network used in this study

## **Reprocessing the miniature mass spectra**

the spec2signal.py includes interpolation, segmentation and normalization of spectra

## **Target resolution mass spectra generation for training** 

the simulate_peaks.py used to generate the mass spectra with target resolution

## **Super-resolution reconstruction of mass spectra**

after training the network, the reconstruct_section.py used to reconstruction the miniature mass spectra

## **Data used in this study**

in the data, segments of the original spectra used for the study are included

in the example-data.zip, the training dataset with simulation labels is included, as well as Brick MS and algorithmic reconstruction results



# Contact 

ab1583564248@gmail.com