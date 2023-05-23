import torch.nn.modules as nn
import torch
import cv2
import numpy as np
import torch.nn.functional as F

# import model_addconv
from model_addconv_repeat import NET
import h5py
import scipy.io as sio
import os


###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def load_setmat(file_path):
    data = h5py.File(file_path)  #

    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)

    pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)

    ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
    ms1 = np.array(ms1, dtype=np.float32) / 2047.0
    ms = torch.from_numpy(ms1)

    gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
    gt1 = np.array(gt1, dtype=np.float32) / 2047.0
    gt = torch.from_numpy(gt1)
    Nn, Wn, Hn, Cn = gt.shape
    ms = ms.permute(0, 3, 1, 2)
    pan = pan.reshape(Nn,Wn,Hn,1).permute(0, 3, 1, 2)
    lms = lms.permute(0, 3, 1, 2)


    return ms, pan, lms, gt, gt1


def load_set(file_path):

    ## ===== case1: NxCxHxW
    # data = h5py.File(file_path)
    # ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
    # shape_size = len(ms1.shape)

    ## ===== case2: HxWxC
    data = sio.loadmat(file_path)  #
    ms1 = data["I_MS_LR"][...]  # NxCxHxW=0,1,2,3
    shape_size = len(ms1.shape)

    if shape_size == 4:  # NxCxHxW
        # tensor type:
        lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        lms = torch.from_numpy(lms1)

        pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0
        pan = torch.from_numpy(pan1)

        ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        ms = torch.from_numpy(ms1)

        return ms, pan, lms

    if shape_size == 3:  # HxWxC
        # tensor type:
        lms1 = data['I_MS'][...]  # HxWxC=0,1,2
        lms1 = np.expand_dims(lms1, axis=0)  # 1xHxWxC
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0 # 1xHxWxC
        lms = torch.from_numpy(lms1).permute(0, 3, 1, 2)  # NxCxHxW  or HxWxC

        pan1 = data['I_PAN'][...]  # HxW
        pan1 = np.expand_dims(pan1, axis=0)  # 1xHxW
        pan1 = np.expand_dims(pan1, axis=3)  # 1xHxWx1
        pan1 = np.array(pan1, dtype=np.float32) / 2047.  # 1xHxWx1
        pan = torch.from_numpy(pan1).permute(0, 3, 1, 2)  # Nx1xHxW:

        ms1 = data['I_MS_LR'][...]  # HxWxC=0,1,2
        ms1 = np.expand_dims(ms1, axis=0)  # 1xHxWxC
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0 # 1xHxWxC
        ms = torch.from_numpy(ms1).permute(0, 3, 1, 2)  # NxCxHxW  or HxWxC

        return ms, pan, lms

###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
# ==============  Main test  ================== #
ckpt = "Weights/wv3/600.pth"   # chose model

def test(file_path):
    # ms, pan, lms = load_set(file_path)

    model = NET().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)
    print()



    with torch.no_grad():

        x1, x2, x3 = testms, testpan, testlms  # read data: CxHxW (numpy type)

        # x1 = x1.cuda().float()  # convert to tensor type:
        x2 = x2.cuda().float()  # convert to tensor type:
        x3 = x3.cuda().float()  # convert to tensor type:

        # x1 = F.interpolate(x1, scale_factor=4, mode="bicubic") # as: x1---> lms

        sr = torch.zeros(x3.size())
        # x1 = F.interpolate(x1, scale_factor=4, mode="bicubic") # as: x1---> lms
        for i in range(x2.size(0)):
            pan1 = x2[i:i + 1, :]
            lms1 = x3[i:i + 1, :]
            # print(lms1.size())
            sr1 = model(lms1, pan1)  # tensor type: sr = 1xCxHxW
            # print(sr1.size())
            # print(sr[i:i+1,:].size())
            sr[i:i + 1, :] = sr1.cpu()
        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = sr.permute(0, 2, 3, 1).detach().numpy()  # to: NxHxWxC


        sr = np.clip(sr, 0, 1)
        print(sr.shape)

        num_exm = sr.shape[0]


        for index in range(num_exm):  # save the DL results to the 03-Comparisons(Matlab)
            file_name = "output_muExm" + str(index) + ".mat"
            file_name2 = "results/WV3/mul"
            save_name = os.path.join(file_name2, file_name )
            sio.savemat(save_name, {'test_result': sr[index, :, :, :]})
            print(index)

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    ##  case1: test on multiple images with the size of NxCxHxW
    # file_path = "/home/office-401-2/Desktop/Machine Learning/Liang-Jian Deng/02-DRPNN-Pytorch/test_data/TestData_wv3.h5"

    ##  case2: test on single image with the size of HxWxC
    # file_path = "Data_Testing/NY1_WV3_RR.mat"
    # file_path='/Data/DataSet/pansharpening/pan_test/test1_mulExm1258.mat'
    file_path = '/Data2/DataSet/pansharpening/pan_test/test1_mulExm1258.mat'
    testms, testpan, testlms,testgt1,_= load_setmat(file_path)

    # file_path = "Data_Testing/NY1_WV3_FR.mat"

    test(file_path)

