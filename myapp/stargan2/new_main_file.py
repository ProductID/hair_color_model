"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse


from torch.backends import cudnn
import torch

from myapp.stargan2.core.new_solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args,src_path,des_path,result_dir):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    assert len(subdirs(args.src_dir)) == 2
    assert len(subdirs(args.ref_dir)) == 2

    solver.sample(src_path,des_path,result_dir)


# def extrat():
#     parser = argparse.ArgumentParser()
#
#     # model arguments
#     parser.add_argument('--img_size', type=int, default=256,
#                         help='Image resolution')
#     parser.add_argument('--num_domains', type=int, default=2,
#                         help='Number of domains')
#     parser.add_argument('--latent_dim', type=int, default=16,
#                         help='Latent vector dimension')
#     parser.add_argument('--hidden_dim', type=int, default=512,
#                         help='Hidden dimension of mapping network')
#     parser.add_argument('--style_dim', type=int, default=64,
#                         help='Style code dimension')
#
#     # weight for objective functions
#     parser.add_argument('--lambda_reg', type=float, default=1,
#                         help='Weight for R1 regularization')
#     parser.add_argument('--lambda_cyc', type=float, default=1,
#                         help='Weight for cyclic consistency loss')
#     parser.add_argument('--lambda_sty', type=float, default=1,
#                         help='Weight for style reconstruction loss')
#     parser.add_argument('--lambda_ds', type=float, default=1,
#                         help='Weight for diversity sensitive loss')
#     parser.add_argument('--ds_iter', type=int, default=100000,
#                         help='Number of iterations to optimize diversity sensitive loss')
#     parser.add_argument('--w_hpf', type=float, default=1,
#                         help='weight for high-pass filtering')
#
#     # training arguments
#     parser.add_argument('--randcrop_prob', type=float, default=0.5,
#                         help='Probabilty of using random-resized cropping')
#     parser.add_argument('--total_iters', type=int, default=100000,
#                         help='Number of total iterations')
#     parser.add_argument('--resume_iter', type=int, default=100000,
#                         help='Iterations to resume training/testing')
#     parser.add_argument('--batch_size', type=int, default=8,
#                         help='Batch size for training')
#     parser.add_argument('--val_batch_size', type=int, default=1,
#                         help='Batch size for validation')
#     parser.add_argument('--lr', type=float, default=1e-4,
#                         help='Learning rate for D, E and G')
#     parser.add_argument('--f_lr', type=float, default=1e-6,
#                         help='Learning rate for F')
#     parser.add_argument('--beta1', type=float, default=0.0,
#                         help='Decay rate for 1st moment of Adam')
#     parser.add_argument('--beta2', type=float, default=0.99,
#                         help='Decay rate for 2nd moment of Adam')
#     parser.add_argument('--weight_decay', type=float, default=1e-4,
#                         help='Weight decay for optimizer')
#     parser.add_argument('--num_outs_per_domain', type=int, default=10,
#                         help='Number of generated images per domain during sampling')
#
#     # misc
#     parser.add_argument('--mode', type=str, default='sample',
#                         choices=['train', 'sample', 'eval', 'align'],
#                         help='This argument is used in solver')
#     parser.add_argument('--num_workers', type=int, default=1,
#                         help='Number of workers used in DataLoader')
#     parser.add_argument('--seed', type=int, default=777,
#                         help='Seed for random number generator')
#
#     # directory for training
#     parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
#                         help='Directory containing training images')
#     parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
#                         help='Directory containing validation images')
#     parser.add_argument('--sample_dir', type=str, default='expr/samples',
#                         help='Directory for saving generated images')
#     parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints/celeba_hq',
#                         help='Directory for saving network checkpoints')
#
#     # directory for calculating metrics
#     parser.add_argument('--eval_dir', type=str, default='expr/eval',
#                         help='Directory for saving metrics, i.e., FID and LPIPS')
#
#     # directory for testing
#     parser.add_argument('--result_dir', type=str, default='expr/results/new',
#                         help='Directory for saving generated images and videos')
#     parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
#                         help='Directory containing input source images')
#     parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
#                         help='Directory containing input reference images')
#     parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
#                         help='input directory when aligning faces')
#     parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
#                         help='output directory when aligning faces')
#
#     # face alignment
#     parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
#     parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')
#
#     # step size
#     parser.add_argument('--print_every', type=int, default=10)
#     parser.add_argument('--sample_every', type=int, default=5000)
#     parser.add_argument('--save_every', type=int, default=10000)
#     parser.add_argument('--eval_every', type=int, default=50000)
#     print(parser)
#     args = parser.parse_args()
#     print('agfjkhsldglkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaas')
#     return args


# src_path = 'assets/representative/celeba_hq/src/female/051340.jpg'
# des_path = 'assets/representative/celeba_hq/ref/female/036619.jpg'
# result_dir = '/home/webtunixhaz/Documents/stargan2/results_all/new.jpg'

def add_with_front_panel(src_path,des_path,result_dir):
    aaa = argparse.Namespace(batch_size=8, beta1=0.0, beta2=0.99,
                             checkpoint_dir='myapp/stargan2/expr/checkpoints/celeba_hq',
                             ds_iter=100000,
                             eval_dir='myapp/stargan2/expr/eval',
                             eval_every=50000, f_lr=1e-06, hidden_dim=512, img_size=256,
                             inp_dir='myapp/stargan2/assets/representative/custom/female',
                             lambda_cyc=1, lambda_ds=1, lambda_reg=1, lambda_sty=1, latent_dim=16,
                             lm_path='myapp/stargan2/expr/checkpoints/celeba_lm_mean.npz',
                             lr=0.0001, mode='sample', num_domains=2, num_outs_per_domain=10, num_workers=1,
                             out_dir='myapp/stargan2/assets/representative/celeba_hq/src/female',
                             print_every=10, randcrop_prob=0.5,
                             ref_dir='myapp/stargan2/assets/representative/celeba_hq/ref',
                             result_dir='myapp/stargan2/results/new',
                             resume_iter=100000,
                             sample_dir='myapp/myapp/stargan2/expr/samples',
                             sample_every=5000, save_every=10000, seed=777,
                             src_dir='myapp/stargan2/assets/representative/celeba_hq/src',
                             style_dim=64, total_iters=100000,
                             train_img_dir='data/celeba_hq/train', val_batch_size=1,
                             val_img_dir='data/celeba_hq/val',
                             w_hpf=1, weight_decay=0.0001,
                             wing_path='myapp/stargan2/expr/checkpoints/wing.ckpt')
    print(aaa)
    print('|||||||||||||||||||||||||||||||||||||||')
    main(aaa,src_path,des_path,result_dir)
# add_with_front_panel(src_path,des_path,result_dir)