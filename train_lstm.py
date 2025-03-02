import os
import sys
import datetime

sys.path.append('./')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from nets.LSTM import Seq2SeqLSTM
from utils.utils_fit import train_lstm_one_epoch
from utils.call_back import LossHistory
from utils.utils import (set_optimizer_lr,get_lr_scheduler,show_config)
from utils.DataLoader import SectionDataSet, Section_dataset_collate

if __name__ == '__main__':
    half_len, num_points_per_mz = 6, 250
    brick_path = './data/Brick/preprocess/single_charge.csv'
    simulate_path = './simulate-data/section.csv'

    input_size, hidden_size, output_size = 250, 128, 250
    seq_len, split_size = half_len*2, num_points_per_mz

    Init_Epoch, Epoch = 0, 200
    batch_size = 64
    Init_lr = 1e-2
    Mini_lr = Init_lr*0.1
    optimizer_type = 'adam'
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 50
    save_dir = 'logs/lstm'
    num_works = 4
    Cuda = True

    model = Seq2SeqLSTM(input_size,hidden_size,output_size,seq_len,split_size)

    mse_loss = nn.MSELoss()

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, [model], input_shape=seq_len*split_size)

    if Cuda:
        cudnn.benchmark = True
        model_train = torch.nn.DataParallel(model)
        model_train = model_train.cuda()

    train_dataset = SectionDataSet(brick_path, simulate_path, half_len, num_points_per_mz)
    num_train = len(train_dataset)
    epoch_step = min(num_train//batch_size, 2000)

    show_config(Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
        Init_lr=Init_lr, Min_lr=Mini_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_works=num_works, num_train=num_train)

    if True:
        optimizer = {
            'adam': optim.Adam(model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(model_train.parameters(), Init_lr, momentum=momentum, nesterov=True)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Mini_lr, Epoch)
        shuffle = True
        train_sampler = None
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_works, pin_memory=True,
                         drop_last=False, collate_fn=Section_dataset_collate, sampler=train_sampler)

        for epoch in range(Init_Epoch, Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            train_lstm_one_epoch(model_train, model, loss_history, optimizer, mse_loss, epoch, epoch_step, gen, Epoch, Cuda,
                                 save_period, save_dir)


