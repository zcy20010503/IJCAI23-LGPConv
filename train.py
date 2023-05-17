
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_wv3_update import Dataset_Pro
import torch.utils.tensorboard as tensorboard
import shutil
from tensorboardX import SummaryWriter
import scipy.io
from model_addconv_repeat import NET
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True 
cudnn.deterministic = True
cudnn.benchmark = False

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0025
epochs = 650
ckpt = 10
batch_size = 32
model_path = "Weights/pannetplus/.pth"
epoch_train_loss_sum = []
epoch_val_loss_sum = []
log_path = "train_log.txt"
# #
# downtime = [10,20,50,200,300,500]
# downtime = [15,30,80,300,500]
# gamma = 1/2
start_epoch = 0
writer = tensorboard.SummaryWriter('./train_logs')

# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = NET().cuda()
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))   ## Load the pretrained Encoder
    print('network is Successfully Loaded from %s' % (model_path))

# summaries(model, grad=True)    ## Summary the Network
criterion = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)   ## optimizer 1: Adam
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)   # learning-srate update

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  ## optimizer 2: SGD
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)  # learning-rate update: lr = lr* 1/gamma for each step_size = 180

# ============= 4) Tensorboard_show + Save_model ==========#
if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs  ## Tensorboard_show: case 1
    shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs



writer = SummaryWriter('./train_logs')    ## Tensorboard_show: case 2


def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/wv3/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,start_epoch=0):
    min_v_loss = 999999
    print('Start training...')
    # epoch 450, 450*550 / 2 = 123750 / 8806 = 14/per imgae

    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, ms, lms, pan = batch[0].cuda(), batch[1].cuda(), \
                                                               batch[2].cuda(), batch[3].cuda()
            optimizer.zero_grad()

            # sr = model(ms, pan, ms_hp, pan_hp, ms_hhp, pan_hhp)  # call model
            # sr = model(ms, pan, ms_hhp, pan_hhp)  # call model1
            sr = model(lms, pan)  # call model for model4_2

            # sr = sr+lms

            loss = criterion(sr, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()  # fixed
            optimizer.step()  # fixed

            # for name, layer in model.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                # writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)

        lr_scheduler.step()  # if update_lr, activate here!
        # if epoch in downtime:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= gamma
        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        # writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss),lr_scheduler.get_lr())  # print loss for each epoch


        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, ms, lms, pan,= batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), \
                                                                   batch[3].cuda(),

                # sr = model(ms, pan, ms_hp, pan_hp, ms_hhp, pan_hhp)  # call model
                sr = model(lms, pan) # call model
                # sr = sr+lms

                loss = criterion(sr, gt)
                epoch_val_loss.append(loss.item())

        # if epoch % 1 == 0:
        #     v_loss = np.nanmean(np.array(epoch_val_loss))
        #     # writer.add_scalar('val/v_loss', v_loss, epoch)
        #     print('             validate loss: {:.7f}'.format(v_loss))

        with open(log_path, mode='a') as filename:
            filename.write('Epoch: {}/{} training loss: {:.7f} '.format(epochs, epoch, t_loss))
            filename.write('\n')
            # print('noise value', model.res_block[0].conv1.weight)

        epoch_train_loss_sum.append(epoch_train_loss[0])
        epoch_val_loss_sum.append(epoch_val_loss[0])
    # writer.close()  # close tensorboard

        if epoch % 5 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss', v_loss, epoch)
            print('             validate loss: {:.7f}'.format(v_loss))

        # if epoch ==1 :
        #     print(model.res_block[0].conv1.weight.type())
        #     weight_1 = model.res_block[0].conv1.weight.cpu()
        #     weight_1=weight_1.detach().numpy()
            # scipy.io.savemat('weight_init.mat',mdict={'weight':weight_1})

        if epoch ==epochs:
            weight_2 = model.res_block[0].conv1.weight.cpu()
            weight_2 = weight_2.detach().numpy()
            scipy.io.savemat('weight_final.mat', mdict={'weight': weight_2})
###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('train.h5')  # creat data for training
   
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('vaild.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
    torch.cuda.empty_cache()
    train(training_data_loader, validate_data_loader)  # call train function
