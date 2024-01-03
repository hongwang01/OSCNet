import os
import os.path
import argparse
import numpy as np
import torch
import time
import h5py
from utils import utils_image
import PIL
from PIL import Image
import utils.save_image as save_img
from network.oscnet import OSCNet
from network.oscnetplus import OSCNetplus

parser = argparse.ArgumentParser(description="OSCNet_Test")
#for model_selection
parser.add_argument('--model', type=str, default="osc", help='osc or oscplus')
parser.add_argument("--model_dir", type=str, default="model_osc/net_latest.pt", help='path to model file')
parser.add_argument("--data_path", type=str, default="data/test/", help='path to test data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str, default="save_results/", help='path to testing results')


#for filter parameterization
parser.add_argument('--padding', type=int, default=4, help='the number of padding during convolution')
parser.add_argument('--inP', type=int, default=5, help='control the basis for filter parameterization')
parser.add_argument('--sizeP', type=int, default=9, help='control the basis for filter parameterization')
parser.add_argument('--ifini', type=float, default=1, help='indicator for filter parameterization')
parser.add_argument('--cdiv', type=float, default=1, help='controlling the updating rate of filter for oscnetplus. For oscnet, it is fixed as 1') # oscnet: default as 1

#for network and dictionary model
parser.add_argument('--num_M', type=int, default=4, help='the number of feature maps at every rotation angle')
parser.add_argument('--num_Q', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--num_rot', type=int, default=8, help='the number of rotation angles')
parser.add_argument('--S', type=int, default=10, help='Stage number S')
parser.add_argument('--T', type=int, default=3, help='Resblocks number in each ProxNet')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating B')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

out_dir = opt.save_path+ opt.model +'/'
mkdir(out_dir)

input_dir = opt.save_path+'/input/'
mkdir(input_dir)

gt_dir = opt.save_path+'/gt/'
mkdir(gt_dir)


def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (maxX - minX)
    return X

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data


test_mask = np.load(os.path.join(opt.data_path, 'testmask.npy'))
def test_image(data_path, imag_idx, mask_idx):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
    gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma= file['ma_CT'][()]
    XLI =file['LI_CT'][()]
    file.close()
    M512 = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    Xma = normalize(Xma, image_get_minmax())  
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    non_mask = 1 - Mask
    return torch.Tensor(Xma).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(non_mask).cuda()
def main():
    # Build model
    print('Loading model ...\n')
    if "plus" not in opt.model:
        net= OSCNet(opt).cuda()
    else:
        net= OSCNetplus(opt).cuda()
    net.eval()
    net.load_state_dict(torch.load(opt.model_dir))
    print_network(net)

    time_test = 0
    count = 0
    for imag_idx in range(200): # for original testing, 200 clean CT images
        print("imag_idx:",imag_idx)
        for mask_idx in range(10): # for original testing, 10 testing metal masks
            Xma, X, XLI, M = test_image(opt.data_path, imag_idx, mask_idx)
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                X0, ListX, ListA = net(Xma, XLI, M)
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
            Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
            Xgtclip = torch.clamp(X / 255.0, 0, 0.5)
            Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
            Xoutnorm = Xoutclip / 0.5
            Xgtnorm = Xgtclip / 0.5
            Xmanorm = Xmaclip / 0.5
            idx = imag_idx *10+ mask_idx  + 1
            Xnorm = [Xoutnorm, Xmanorm, Xgtnorm]
            dir = [out_dir, input_dir, gt_dir]
            save_img.imwrite(idx, dir, Xnorm)
            print('Times: ', dur_time)
            count += 1
    print(100*'*')
if __name__ == "__main__":
    main()

