import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from core.model import build_model
import torchvision.transforms.functional as TF
from torchvision import transforms

# newsize = (256, 256)
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# image_src = Image.open('assets/representative/celeba_hq/ref/female/036619.jpg').resize(newsize)
# # x_src = TF.to_tensor(image_src).to(device)
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
# x_src=image_src
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])
# x_src=transform(x_src)
# x_src.unsqueeze_(0)
#
# print(x_src.shape,'shape1')
# print( torch.squeeze(x_src).shape)
# # x_src=src.x
# # print(type(x_src))
# # print(x_src.shape)
# surce_image=x_src
# surce_image =torch.squeeze(surce_image).cpu().detach().numpy()
# plt.imshow(np.transpose(surce_image, (1, 2, 0)))
# plt.show()
# io=torch.tensor([0]).cuda()
# print(io)

nets_ema = build_model(args)



module_dict = torch.load('expr/checkpoints/celeba_hq/100000_nets_ema.ckpt')
for name, module in module_dict.items():
    module.load_state_dict(module_dict[name])
'''
 nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        print('started next'*90)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))

        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))
        print(type(ref))

        fname = join(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        # utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)
        nets=nets_ema
        from PIL import Image
        import torchvision.transforms.functional as TF
        newsize = (256, 256)

        image_src = Image.open('assets/representative/celeba_hq/src/female/051340.jpg').resize(newsize)
        x_src = TF.to_tensor(image_src)
        x_src.unsqueeze_(0).cuda()
        print(x_src.shape)
        # x_src=src.x
        # print(type(x_src))
        # print(x_src.shape)
        surce_image=x_src
        surce_image =np.squeeze(surce_image.cpu().detach().numpy())
        plt.imshow(np.transpose(surce_image, (1, 2, 0)))
        plt.show()


        print(type(x_src), 'source type')
        y_ref=ref.y
        print(type(y_ref))
        print(y_ref.shape)
        image_ref = Image.open('assets/representative/celeba_hq/ref/female/036619.jpg').resize(newsize)
        x_ref = TF.to_tensor(image_ref)
        x_ref.unsqueeze_(0).cuda()
        print(x_ref.shape,'shape of tensor')

        # x_ref=ref.x
        # print(type(x_ref))
        # print(x_ref.shape)
        rurce_image = np.squeeze(x_ref.cpu().detach().numpy())
        plt.imshow(np.transpose(rurce_image, (1, 2, 0)))
        plt.show()

        N, C, H, W = x_src.size()
        wb = torch.ones(1, C, H, W).to(x_src.device)
        x_src_with_wb = torch.cat([wb, x_src], dim=0)

        masks = nets.fan.get_heatmap(x_src.cuda()) if args.w_hpf > 0 else None
        s_ref = nets.style_encoder(x_ref.cuda(), y_ref)
        print(type(s_ref),'after style')
        print(s_ref.shape,'shape of style')

        # s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
        # x_concat = [x_src_with_wb]
        # for i, s_ref in enumerate(s_ref_list):
        # new_file = str(i) + '.jpg'
        x_fake = nets.generator(x_src.cuda(), s_ref, masks=masks)
        # x_fake=denormalize(x_fake)
        print(type(x_fake.cpu().detach().numpy()), '+' * 90)
        # x_fake_with_ref = torch.cat([x_ref[i:i + 1], x_fake], dim=0)
        # im_src = np.squeeze(denormalize(x_src).cpu().detach().numpy())
        # im_ref =np.squeeze(denormalize(s_ref).cpu().detach().numpy())

        ndarr =  torch.squeeze(denormalize(x_fake)).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.show()
        im_fake =np.squeeze(denormalize(x_fake).to('cpu', torch.uint8).detach().numpy())
        # new_image=s_ref.cpu().detach().numpy()
        # print(new_image.shape,'p'*9)
        # reshaped=np.squeeze(s_ref.cpu().detach().numpy())
        # print(im_src.shape,'o'*99)
        #
        # # plt.figure()
        #
        # # subplot(r,c) provide the no. of rows and columns
        # f, axarr = plt.subplots(2, 1)
        # #
        # # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        # # axarr[0].imshow(np.reshape(im_src, (256,256,3)))
        # # # np.reshape(im_fake, (256,256,3))
        # # # axarr[1].imshow(np.reshape(im_ref, (256,256,3)))
        plt.imshow(ndarr)
        plt.show()

'''

'''
"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join
import time
import datetime
from munch import Munch
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules as mod
from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        # for name, module in self.nets.items():
        #     utils.print_network(module, name)
        #     setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)


        self.ckptios = [CheckpointIO(join(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

        # nn.Module.to(self.device)
        # nn.Module.named_children()
        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _load_checkpoint(self, step):
        count=1
        for ckptio in self.ckptios:
            print(count)
            count=count+1
            ckptio.load(step)

    # def _reset_grad(self):
    #     for optim in self.optims.values():
    #         optim.zero_grad()



    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        # if torch.cuda.is_available():
        #     self.nets_ema.cuda()
        #     self.nets.cuda()
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        print('started next'*90)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))

        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))
        print(type(ref))

        fname = join(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        # utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)
        nets=nets_ema
        x_src=src.x
        print(type(x_src))
        print(x_src.shape)
        surce_image=x_src
        surce_image =np.squeeze(surce_image.cpu().detach().numpy())
        plt.imshow(np.transpose(surce_image,(1,2,0)))
        plt.show()


        print(type(x_src), 'source type')
        y_ref=ref.y
        print(type(y_ref))
        print(y_ref.shape)

        x_ref=ref.x
        print(type(x_ref))
        print(x_ref)
        surce_image = x_ref
        surce_image = np.squeeze(surce_image.cpu().detach().numpy())
        plt.imshow(np.transpose(surce_image, (1, 2, 0)))
        plt.show()

        N, C, H, W = x_src.size()
        wb = torch.ones(1, C, H, W).to(x_src.device)
        x_src_with_wb = torch.cat([wb, x_src], dim=0)

        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
        s_ref = nets.style_encoder(x_ref, y_ref)
        print(type(s_ref),'after style')
        print(s_ref.shape,'shape of style')

        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
        x_concat = [x_src_with_wb]
        # for i, s_ref in enumerate(s_ref_list):
        # new_file = str(i) + '.jpg'
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        # x_fake=denormalize(x_fake)
        print(type(x_fake.cpu().detach().numpy()), '+' * 90)
        # x_fake_with_ref = torch.cat([x_ref[i:i + 1], x_fake], dim=0)
        # im_src = np.squeeze(denormalize(x_src).cpu().detach().numpy())
        # im_ref =np.squeeze(denormalize(s_ref).cpu().detach().numpy())
        im_fake =np.squeeze(denormalize(x_fake).cpu().detach().numpy())
        # new_image=s_ref.cpu().detach().numpy()
        # print(new_image.shape,'p'*9)
        # reshaped=np.squeeze(s_ref.cpu().detach().numpy())
        # print(im_src.shape,'o'*99)
        #
        # # plt.figure()
        #
        # # subplot(r,c) provide the no. of rows and columns
        # f, axarr = plt.subplots(2, 1)
        # #
        # # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        # # axarr[0].imshow(np.reshape(im_src, (256,256,3)))
        # # # np.reshape(im_fake, (256,256,3))
        # # # axarr[1].imshow(np.reshape(im_ref, (256,256,3)))
        plt.imshow(np.transpose(im_fake,(1,2,0)))
        plt.show()

            # x_concat += [x_fake_with_ref]
            # save_image(x_concat,1, new_file)

        # quit()
        # x_concat = torch.cat(x_concat, dim=0)
        # save_image(x_concat, N + 1, filename)
        #
        # fname = join(args.result_dir, 'video_ref.mp4')
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

'''