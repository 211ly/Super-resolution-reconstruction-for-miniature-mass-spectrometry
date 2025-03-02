import torch

from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
import numpy as np


def kl_divergence(mu,logvar):
    kl_div = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div

def train_VAE_one_epoch(VAE_train,VAE,loss_History,optimizer,MSE_loss,
                        epoch, epoch_step, Epoch, gen,
                        save_period, save_dir, cuda=True):
    """
    :param VAE_train: model to train
    :param VAE: trained model to save
    :param loss_History: write into tensorboard
    :param epoch: total epoch
    :param epoch_step: the max batch for one epoch
    :param Epoch: current epoch
    :param gen: DataLoader
    :param cuda: use the cuda
    :param save_period: how many epochs to save
    :param save_dir: the dir to save model
    :return:None
    """
    total_loss, total_mse_loss, total_kl_loss = 0., 0., 0.

    print("Start Train\n")
    pbar = tqdm(total=epoch_step, desc=f'Epoch{epoch+1}/{Epoch}', postfix=dict, mininterval=0.3)

    for i, batch in enumerate(gen):
        if i >= epoch_step:
            break

        X = batch[0]


        with torch.no_grad():
            if cuda:
                X = X.cuda()

        optimizer.zero_grad()

        output, mu, logvar = VAE_train(X)

        mse_loss, kl_loss = MSE_loss(output, X), kl_divergence(mu, logvar)
        loss = mse_loss + 1000 * kl_loss

        loss.backward()
        optimizer.step()

        total_kl_loss += 1000 * kl_loss.item()
        total_mse_loss += mse_loss.item()
        total_loss += loss.item()
        pbar.set_postfix(**{'loss': total_loss/(i+1), 'mse_loss': total_mse_loss/(i+1), 'kl_loss':total_kl_loss/(i+1)})
        pbar.update(1)

    total_loss = total_loss/epoch_step
    total_mse_loss /= epoch_step
    total_kl_loss /= epoch_step

    pbar.close()
    print('Epoch:' + str(epoch+1) + '/' + str(Epoch))
    print('Loss:%.4f' % total_loss)
    loss_History.append_loss(epoch+1,total_loss=total_loss,total_mse_loss=total_mse_loss,total_kl_loss=total_kl_loss)

    if (epoch+1) % save_period == 0 or epoch+1 == Epoch:
        torch.save(VAE.state_dict(), os.path.join(save_dir,'Epoch%d-Loss%.4f.pth'%(epoch+1,total_loss)))


    torch.save(VAE.state_dict(), os.path.join('last_epoch.pth'))




def train_val_AE_one_epoch(AE_train,
                           AE,
                           loss_History,
                           optimizer,
                           MSE_loss,
                           epoch,
                           epoch_step, Epoch, train_gen,val_gen,
                           save_period, save_dir,
                           min_val_loss,
                           cuda=True):

    train_loss, val_loss = 0., 0.
    print("Start Train\n")
    pbar = tqdm(total=epoch_step, desc=f'Epoch{epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for i, batch in enumerate(train_gen):
        if i >= epoch_step:
            break

        X = batch[0]

        with torch.no_grad():
            if cuda:
                X = X.cuda()

        optimizer.zero_grad()

        output = AE_train(X)

        loss = MSE_loss(output, X)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(
            **{'loss': train_loss / (i + 1)})
        pbar.update(1)

    train_loss = train_loss / epoch_step
    pbar.close()

    print("Start Validate\n")

    AE.eval()
    with torch.no_grad():
        for input, _ in val_gen:
            input = input.cuda()
            output = AE(input)
            val_loss += torch.nn.functional.mse_loss(output, input, reduction='sum').item()
    val_loss /= len(val_gen)


    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Train_loss:%.4f Val_loss:%.4f' % (train_loss, val_loss))
    loss_History.append_loss(epoch + 1, train_loss=train_loss,val_loss=val_loss)

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(AE.state_dict(), os.path.join(save_dir, 'Epoch%d-Loss%.4f.pth' % (epoch + 1,train_loss)))

    if val_loss < min_val_loss:
        torch.save(AE.state_dict(),os.path.join(save_dir,'best.pth'))

    return val_loss

def train_AE_one_epoch(AE_train,
                           AE,
                           loss_History,
                           optimizer,
                           MSE_loss,
                           epoch,
                           epoch_step, Epoch, train_gen,
                           save_period, save_dir,
                           cuda=True):

    train_loss = 0.
    print("Start Train\n")
    pbar = tqdm(total=epoch_step, desc=f'Epoch{epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for i, batch in enumerate(train_gen):
        if i >= epoch_step:
            break

        X = batch[0]

        with torch.no_grad():
            if cuda:
                X = X.cuda()

        optimizer.zero_grad()

        output = AE_train(X)

        loss = MSE_loss(output, X)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(
            **{'loss': train_loss / (i + 1)})
        pbar.update(1)

    train_loss = train_loss / epoch_step
    pbar.close()

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Train_loss:%.4f' % train_loss)
    loss_History.append_loss(epoch + 1, train_loss=train_loss)

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(AE.state_dict(), os.path.join(save_dir, 'Epoch%d-Loss%.4f.pth' % (epoch + 1,train_loss)))

def output_latent(input_data, ae_model):

    with torch.no_grad():
        latent = ae_model.get_latent(input_data)

    return latent


def train_regressor_one_epoch(re_train,
                              re, ae,
                              loss_History,
                              optimizer,
                              MSE_loss,
                              epoch,
                              epoch_step, Epoch, train_gen,
                              save_period, save_dir,
                              cuda=True):
    train_loss = 0.
    print("Start Train\n")
    pbar = tqdm(total=epoch_step, desc=f'Epoch{epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for i, batch in enumerate(train_gen):
        if i >= epoch_step:
            break

        X,y = batch[0],batch[1]


        with torch.no_grad():
            if cuda:
                X = X.cuda()
                y = y.cuda()


        optimizer.zero_grad()

        output = re_train(X)
        y = output_latent(y, ae)
        loss = MSE_loss(output, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(
            **{'loss': train_loss / (i + 1)})
        pbar.update(1)

    train_loss = train_loss / epoch_step
    pbar.close()

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Train_loss:%.4f' % train_loss)
    loss_History.append_loss(epoch + 1, train_loss=train_loss)

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(re.state_dict(), os.path.join(save_dir, 'Epoch%d-Loss%.4f.pth' % (epoch + 1, train_loss)))
    return train_loss

def train_lstm_one_epoch(net_train, net,
                         loss_history, optimizer, mse_loss,
                         epoch, epoch_step, gen, Epoch, cuda, save_period, save_dir):
    total_loss = 0.

    print("Start Train\n")
    pbar = tqdm(total=epoch_step, desc=f'Epoch{epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for i, batch in enumerate(gen):
        if i >= epoch_step:
            break
        lr, hr = batch
        lr = lr.unsqueeze(1)
        hr = hr.unsqueeze(1)

        with torch.no_grad():
            if cuda:
                lr, hr = lr.cuda(0), hr.cuda(0)

        optimizer.zero_grad()

        output = net_train(lr)
        loss = mse_loss(output, hr)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(
            **{'loss': loss / (i + 1)})
        pbar.update(1)

    total_loss = total_loss/epoch_step
    pbar.close()
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Loss: %.4f' % total_loss)
    loss_history.append_loss(epoch + 1, total_loss=total_loss)

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'Epoch%d-Loss%.4f.pth' % (
            epoch + 1, total_loss)))




