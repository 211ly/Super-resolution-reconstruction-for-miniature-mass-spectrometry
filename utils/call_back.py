import os
import matplotlib
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

        try:
            for m in model:
                dummy_input = torch.randn(*input_shape)
                self.writer.add_graph(m, dummy_input)
        except:
            pass

    def append_loss(self, epoch, **kwargs):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        print(kwargs)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, [])
            getattr(self, key).append(value)
            with open(os.path.join(self.log_dir, key + ".txt"), 'a') as f:
                f.write(str(value))
                f.write("\n")
            self.writer.add_scalar(key, value, epoch)
        self.loss_plot(**kwargs)

    def loss_plot(self, **kwargs):
        """Dynamic plot the loss."""
        plt.figure()
        for key, value in kwargs.items():
            losses = getattr(self, key)
            plt.plot(range(len(losses)), losses, linewidth=2, label=key)

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, 'epoch_loss.png'))

        plt.cla()
        plt.close("all")