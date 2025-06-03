import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.rockts import RockTS
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls
import random
import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset', type=str, default='ETTh1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=0, help='whether to use time features or not')
# Patch
parser.add_argument('--patch_len', type=int, default=48, help='patch length')
parser.add_argument('--stride', type=int, default=48, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='robust_prob ', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=1, help='training the model')

parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--cost_lambda', type=float, default=1, help='cost lambda')
parser.add_argument('--r', type=float, default=0.9, help='cost lambda')

parser.add_argument('--cuda', type=str, default='3', help='cuda number')
parser.add_argument('--head_first', type=int, default=1, help='')

args = parser.parse_args()
print('args:', args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.environ['CUDA_VISIBLE_DEVICES']=args.cuda

args.save_model_name = 'rockts_supervised'+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset + '/rockts_supervised/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

def get_model(c_in, args):
    """
    c_in: number of input variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = RockTS(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,                
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='prediction',
                res_attention=False
                )    
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # define learner
    learn = Learner(dls, model, loss_func, cbs=cbs)                        
    # fit the data to the model
    return learn.lr_finder()


def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    print('in out', dls.vars, dls.c, dls.len)
    
    # get model
    model = get_model(dls.vars, args)

    # get loss
    # loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.L1Loss(reduction='mean')

    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path )
        ]

    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse],
                        cost_lambda = args.cost_lambda,
                        r = args.r,
                        )
                        
    # fit the data to the model
    if args.head_first:
        learn.head_first_train(n_epochs=args.n_epochs, base_lr=lr, freeze_epochs=20, pct_start=0.2)
    else:
        learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    weight_path = args.save_path + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])         # out: a list of [pred, targ, score_values]
    return out


if __name__ == '__main__':

    if args.is_train:   # training mode
        train_func(args.lr)
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
    else:   # testing mode
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
   
    print('----------- Complete! -----------')


