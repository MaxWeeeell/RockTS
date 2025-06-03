
__all__ = ['RockTS']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *

from ..models.maskgen import MaskGenerator
# from tint.models import MLP, RNN


            
# Cell
class RockTS(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False, **kwargs):

        super().__init__()

        self.bottleneck = Encoder(c_in, num_patch=num_patch*patch_len, patch_len=1, 
                                n_layers=1, d_model=16, n_heads=1, 
                                shared_embedding=shared_embedding, d_ff=32,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        self.backbone_rec = Encoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        self.backbone = Encoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        
        self.use_ste = True  #gumbel softmax hard?
        self.mask_generator = MaskGenerator(d_z = num_patch*patch_len, d_model = 16, d_pe = 1, use_ste = self.use_ste)
        # self.mask_connection_src = MLP([2, 32, 1], activations='elu', dropout=0.0)
        self.ot = OT(fealen = num_patch*patch_len)

        # Head
        self.n_vars = c_in
        self.d_model = d_model
        self.num_patch = num_patch
        self.patch_len = patch_len

        self.head_rec = PretrainHead(d_model, patch_len, head_dropout)
        self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)


    def forward(self, z):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   
        bs, num_patch, n_vars, patch_len = z.shape
        z = z.permute(0,1,3,2).reshape(bs, -1, n_vars)    # z: [bs x seq_len x nvars]
        z = z.unsqueeze(-1) # z: [bs x seq_len x nvars x  1]

        z_rep_regular, z_regular, bottleneck_atten = self.bottleneck(z)             # z: [bs x nvars x d_model x num_patch]  # matrix: [bs*nvars x 1 x num_patch x num_patch]

        matrix_prob, matrix_mask = self.mask_generator(z_rep_regular.permute(0,1,3,2), z_regular.permute(0,1,3,2))
        mask_num = matrix_mask.numel()-matrix_mask.sum()

        z = z.squeeze(-1).reshape(bs, num_patch, patch_len, n_vars).permute(0,1,3,2)  # z: tensor [bs x num_patch x n_vars x patch_len]
        masked_z = self.multivariate_mask(z, matrix_mask)

        masked_z_rep,_,_ = self.backbone_rec(masked_z) 
        rec_z= self.head_rec(masked_z_rep) 
        # rec_z= rec_z*(1-matrix_mask.reshape(-1, self.n_vars, self.num_patch, self.patch_len).transpose(1,2))+masked_z 

        # ot_z, cost = self.ot(rec_z, bottleneck_atten, matrix_mask.reshape(-1, self.num_patch*self.patch_len)) 
        ot_z, cost = self.ot(rec_z, bottleneck_atten, (1-matrix_prob)*(1-matrix_mask).reshape(-1, self.num_patch*self.patch_len)) 


        rec_z_rep,_,_ = self.backbone(ot_z*(1.0-matrix_mask.reshape(-1, self.n_vars, self.num_patch, self.patch_len).transpose(1,2)) + masked_z)    # z: [bs x nvars x d_model x num_patch] 
        pred_mask = self.head(rec_z_rep) 

        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        # return pred_mask, rec_z, (ste_mask.numel()-mask_sum)/ste_mask.numel()
        
        return pred_mask, z, masked_z, rec_z, ot_z, matrix_prob, matrix_mask, cost
 
    def multivariate_mask(self, src, matrix_mask):
        # First apply mask directly on input:
        baseline = self._get_baseline(src)
        # baseline = src

        ste_mask_rs = matrix_mask.reshape(-1, self.n_vars, self.num_patch, self.patch_len).transpose(1,2)
        # if len(ste_mask_rs.shape) == 2:
        #     ste_mask_rs = ste_mask_rs.unsqueeze(-1)


        # src_masked_ref = src * ste_mask_rs + (1 - ste_mask_rs) * baseline#self.baseline_net(src)#baseline\
        src_masked_ref = src * ste_mask_rs
        
        # src_masked = self.mask_connection_src(torch.stack([src * ste_mask_rs, (1 - ste_mask_rs) * baseline], dim=-1)).squeeze(-1)

        # src_masked = self.mask_connection_src(torch.stack([src, ste_mask_rs], dim=-1)).squeeze(-1)
        # src_masked = src_masked_ref

        return src_masked_ref
    
    def _get_baseline(self, src):

        # mu =  torch.mean(src,dim=(0,1,3))
        # std = torch.std(src, dim=(0,1,3))
        # samp = torch.stack([torch.normal(mean = mu[i], std = std[i], size=src[:,:,0,:].shape).to('cuda') for i in range(src.shape[2])], dim = 2)

        src_pad = src.permute(0,1,3,2).reshape(-1,self.num_patch*self.patch_len, self.n_vars)
        padding1 = (0, 0,1,0)  # 在 T 维度的开头添加 1 个时间步的 padding
        padding2 = (0, 0,0,1)  # 在 T 维度的开头添加 1 个时间步的 padding

        # 在 T 维度上添加 padding，使用 'replicate' 模式复制边缘值
        left_padded = F.pad(src_pad, padding1, mode='replicate')
        right_padded = F.pad(src_pad, padding2, mode='replicate')
        samp = (left_padded[:,:-1,:] + right_padded[:,1:,:])/2.0
        samp = samp.reshape(-1, self.num_patch, self.patch_len, self.n_vars).permute(0, 1, 3, 2)

        return samp

class OT(nn.Module):
    def __init__(self,fealen):
        super(OT,self).__init__()
        # self.P=nn.Parameter(torch.diag(torch.ones(fealen)))
        # self.device=pars.device
        self.fealen=fealen
        self.linear_p = nn.Linear(fealen, fealen)
        self.softmax=nn.Softmax(dim=1)
        self.norm_attn = nn.LayerNorm(fealen)
        # if pars.omit:
        #     self.cost=pars.exeTime[1:]
        # else:
        #     self.cost=pars.exeTime

    def getC(self):
        intervals = torch.tensor(self.cost,dtype=torch.float)
        intervals/=intervals.sum()
        X, Y = torch.meshgrid(intervals, intervals)
        C=X-Y
        mask=C<0.
        C[mask]=0.
        C=C.to(self.device)
        return C

    def forward(self, x, P, matrix_mask):#x:batch,winlen,fealen
        bs, num_patch, n_vars, patch_len = x.shape
        # C = (1-matrix_mask).unsqueeze(-2).expand(-1, self.fealen, -1)
        C = matrix_mask.unsqueeze(-2).expand(-1, self.fealen, -1)
        P = P.squeeze()
        P = self.linear_p(P)
        P = self.norm_attn(P)
        P = self.softmax(P.reshape(-1, self.fealen)).reshape(-1, self.fealen, self.fealen)   #按行算
        # P = self.softmax(P.permute(0,2,1).reshape(-1, self.fealen)).reshape(-1, self.fealen, self.fealen).permute(0,2,1)       # [bs*n_var  x  seq_len  x seq_len]
        # print(torch.max(P[0,0,:]))
        # print(torch.min(P[0,0,:]))
        x = x.permute(0,2,1,3).reshape(-1,1,self.fealen)        # [bs*n_var  x  1  x seq_len]
        tx=torch.matmul(x,P).squeeze()      # [bs*n_var  x  numpatch*patch_len]
        tx = tx.reshape(bs, n_vars, num_patch, patch_len).permute(0,2,1,3)  # [bs x numpatch x n_var x patch_len]
        x=x.squeeze()    # [bs*n_var  x seq_len]
        x=x.repeat(1, self.fealen).reshape(-1, self.fealen, self.fealen)
        # cost=torch.mean(P.transpose(1,2)*C*torch.abs(x-torch.mean(x)))
        # cost=torch.mean(P.transpose(1,2)*C*torch.abs(x))
        # cost=torch.mean(P.transpose(1,2)*C*torch.abs(x))
        cost=torch.mean(P*C) #弗罗贝纽斯点积
        return tx, cost  


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)


    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class Encoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)
        
        

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z, atten = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]

        return z, x.transpose(2,3) , atten
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, atten, scores = mod(output, prev=scores)
            return output, atten
        else:
            for mod in self.layers: output, atten = mod(output)
            return output, atten



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, attn, scores
        else:
            return src, attn



