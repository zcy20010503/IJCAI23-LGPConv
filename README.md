# LGPConv-Learnable-Gaussian-Perturbation-Convolution-for-Lightweight-Pansharpening
* This is code for paper: " LGPConv-Learnable-Gaussian-Perturbation-Convolution-for-Lightweight-Pansharpening", accepted by IJCAI 2023
* Homepage for the corresponding author: Prof. Liang-Jian Deng (https://liangjiandeng.github.io/)

## Training Platform
* Python 3.8 (Recommend to use Anaconda)
* Pytorch 1.8.0
* NVIDIA GPU + CUDA
* Python packages: pip install numpy scipy h5py
* TensorBoard

## Dataset and Toolbox
The datasets used in this paper is WorldView-3, QuickBird and GaoFen-2. And you can find these three dataset and useful tools in https://liangjiandeng.github.io/PanCollection.html.

## Train and Test 
Training and testing codes are in the current folder.

* The code for training is in train.py, and we also provide our pretrained model "650.pth".

* For training, you need to set the file_path in the main function, adopt to your train set, validate set, and test set as well. Our code train the .h5 file, you may change it through changing the code in main function.

## Concat
We are glad to hear from you. If you have any questions, please feel free to contact chenyuzhaouestc@gmail.com.

