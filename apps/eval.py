import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path, mask_path, param_path):
        # Name
        img_name = []
        for i in range(len(image_path)):
            img_name.append(os.path.splitext(os.path.basename(image_path[i]))[0])
        # img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        projection_matrix_90 = np.identity(4)
        projection_matrix_90[0, 0] = 0
        projection_matrix_90[0, 2] = 1
        projection_matrix_90[1, 1] = -1
        projection_matrix_90[2, 0] = -1
        projection_matrix_90[2, 2] = 0
        calib_90 = torch.Tensor(projection_matrix_90).float()
        # Mask
        mask = []
        for i in range(len(mask_path)):
            t_mask = Image.open(mask_path[i]).convert('L')
            t_mask = transforms.Resize(self.load_size)(t_mask)
            t_mask = transforms.ToTensor()(t_mask).float()
            mask.append(t_mask)

        mask_result = torch.stack(mask, dim=0)
        # mask_result = mask_result.view(mask_result[0] * mask_result[1], mask_result[2], mask_result[3], mask_result[4])
        # image
        image = []
        for i in range(len(image_path)):
            t_image = Image.open(image_path[i]).convert('RGB')
            t_image = self.to_tensor(t_image)
            t_image = t_image
            t_image = mask[i].expand_as(t_image) * t_image
            image.append(t_image)
        image_result = torch.stack(image, dim=0)
        # image_result = image_result.view(image_result[0] * image_result[1], image_result[2], image_result[3], image_result[4])

        # #calib
        # calib_list = []
        # for i in range(len(param_path)):
        #     # loading calibration data
        #     param = np.load(param_path[i], allow_pickle=True)
        #     # pixel unit / world unit
        #     ortho_ratio = param.item().get('ortho_ratio')
        # #     # world unit / model unit
        #     scale = param.item().get('scale')
        # #     # camera center world coordinate
        #     center = param.item().get('center')
        # #     # model rotation
        #     R = param.item().get('R')
        # #
        #     translate = -np.matmul(R, center).reshape(3, 1)
        #     extrinsic = np.concatenate([R, translate], axis=1)
        #     extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        #     # Match camera space to image pixel space
        #     scale_intrinsic = np.identity(4)
        #     scale_intrinsic[0, 0] = scale / ortho_ratio
        #     scale_intrinsic[1, 1] = -scale / ortho_ratio
        #     scale_intrinsic[2, 2] = scale / ortho_ratio
        #     # Match image pixel space to image uv space
        #     uv_intrinsic = np.identity(4)
        #     uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
        #     uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
        #     uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
        #     # Transform under image pixel space
        #     trans_intrinsic = np.identity(4)
        #
        #     intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        #     calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        #     calib_list.append(calib)
        #
        calib_result = torch.stack([calib, calib_90], dim=0)
        # calib_result = calib_result.view(calib_result[0] * calib_result[1], calib_result[2], calib_result[3])
        return {
            'name': img_name,
            'img': image_result,
            'calib': calib_result,
            'mask': mask_result,
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]
    test_params = [f[:-4] + '_param.npy' for f in test_images]

    print("num; ", len(test_masks))

    try:
        print(test_images, test_masks)
        data = evaluator.load_image(test_images, test_masks, test_params)
        evaluator.eval(data, True)
    except Exception as e:
        print("error:", e.args)

    # for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
    #     try:
    #         print(image_path, mask_path)
    #         data = evaluator.load_image(image_path, mask_path)
    #         evaluator.eval(data, True)
    #     except Exception as e:
    #        print("error:", e.args)
