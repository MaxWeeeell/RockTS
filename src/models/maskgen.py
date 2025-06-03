import torch
import math
from torch import nn
import torch.nn.functional as F
from src.models.layers.decoder_cnn import Decoder

# from txai.models.encoders.transformer_simple import TransformerMVTS
# from txai.smoother import smoother, exponential_smoother
# from txai.utils.functional import transform_to_attn_mask
# from txai.models.encoders.positional_enc import PositionalEncodingTF
# from tint.models import MLP, RNN

trans_decoder_default_args = {
    "nhead": 1, 
    "dim_feedforward": 32, 
}

MAX = 10000.0

class MaskGenerator(nn.Module):
    def __init__(self, 
            d_z, 
            # max_len,
            d_pe = 16,
            trend_smoother = False,
            agg = 'max',
            pre_agg_mlp_d_z = 32,
            time_net_d_z = 64,
            trans_dec_args = trans_decoder_default_args,
            n_dec_layers = 2,
            tau = 1.0,
            use_ste = True,
            d_model = 128
        ):
        super(MaskGenerator, self).__init__()

        self.d_z = d_z
        self.pre_agg_mlp_d_z = pre_agg_mlp_d_z
        self.time_net_d_z = time_net_d_z
        self.agg = agg
        # self.max_len = max_len
        self.trend_smoother = trend_smoother
        self.tau = tau
        self.use_ste = use_ste

        # self.d_inp = self.d_z - d_pe
        self.d_inp = d_pe

        # dec_layer = nn.TransformerDecoderLayer(d_model = d_z, **trans_dec_args) 
        self.mask_decoder =  Decoder(d_layers = 1, patch_len=1, d_model=d_model, n_heads=1, d_ff=32)

        # self.mask_decoder = nn.TransformerDecoder(dec_layer, num_layers = n_dec_layers)

        # self.mask_decoder = nn.Sequential(
        #         RNN(
        #             input_size=d_z,
        #             rnn="gru",
        #             hidden_size=d_z,
        #             bidirectional=True,
        #         ),
        #         MLP([2 * d_z, d_z]),
        #     )

        
        self.pre_agg_net = nn.Sequential(
            nn.Linear(d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
            nn.Linear(self.pre_agg_mlp_d_z, self.pre_agg_mlp_d_z),
            nn.PReLU(),
        )

        if self.d_inp > 1:
            self.time_prob_net = nn.Sequential(nn.Linear(d_model, self.d_inp), nn.Sigmoid())
        else:
            self.time_prob_net = nn.Linear(d_model, 2)

        # if self.d_inp > 1:
        #     self.time_prob_net =  MLP([d_z, 64, self.d_inp])
        #     self.time_prob_net_std = MLP([d_z, 64, self.d_inp])

        # else:
        #     self.time_prob_net =  MLP([d_z, 64, 2])
        #     self.time_prob_net_std = MLP([d_z, 64, 2])


        # self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        # self.init_weights()

    # def init_weights(self):
    #     def iweights(m):
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform(m.weight)
    #             m.bias.data.fill_(0.01)

    #     self.time_prob_net.apply(iweights)
    #     self.pre_agg_net.apply(iweights)

    def reparameterize(self, total_mask):

        if self.d_inp == 1:
            if total_mask.shape[-1] == 1:
                # Need to add extra dim:
                inv_probs = 1 - total_mask
                total_mask_prob = torch.cat([inv_probs, total_mask], dim=-1)
            else:
                total_mask_prob = total_mask.softmax(dim=-1)
        else:
            # Need to add extra dim:
            inv_probs = 1 - total_mask
            total_mask_prob = torch.stack([inv_probs, total_mask], dim=-1)

        total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau = self.tau, hard = self.use_ste)[...,1]

        return total_mask_prob[...,1], total_mask_reparameterize

    def forward(self, z_seq, src, times = None, get_agg_z = False):
        # x = torch.cat([src, self.pos_encoder(times)], dim = -1) # t bs n
        x = src

        # x = x.transpose(1,0) ###########


        # if torch.any(times < -1e5):
        #     tgt_mask = (times < -1e5).transpose(0,1)
        # else:
        #     tgt_mask = None

        z_seq_dec = self.mask_decoder(x, z_seq)

        # z_seq_dec = self.mask_decoder(x)  ###########
        # z_seq_dec = z_seq_dec.transpose(1,0)  ###########


        z_seq_dec = z_seq_dec.reshape(-1, z_seq_dec.shape[2], z_seq_dec.shape[3]).permute(0,2,1)
        
        z_pre_agg = self.pre_agg_net(z_seq_dec)   #  B*C x D x N

        # mean = self.time_prob_net(z_seq_dec)
        # std = self.time_prob_net_std(z_seq_dec)
        # p_time = self.gauss_sample(mean, std)

        p_time = self.time_prob_net(z_seq_dec.permute(0,2,1))    #  B*C x D x  seq_len

        # total_mask_reparameterize = self.reparameterize(p_time.transpose(0,1))
        total_mask_prob, total_mask_reparameterize = self.reparameterize(p_time)
        # if self.d_inp == 1:
        #     total_mask = p_time.transpose(0,1).softmax(dim=-1)[...,1].unsqueeze(-1)
        # else:
        #     total_mask = p_time.transpose(0,1) # Already sigmoid transformed

        # Transpose both src and times below bc expecting batch-first input

        # TODO: Get time and cycle returns later
        # total_mask = total_mask.numel()-total_mask_reparameterize.sum()

        if get_agg_z:
            agg_z = z_pre_agg.max(dim=0)[0]
            return total_mask_prob, total_mask_reparameterize, agg_z
        else:
            return total_mask_prob, total_mask_reparameterize
        
    # def gauss_sample(self, mean_logit, std, training=True):
    #     if training:
    #         att_bern = (mean_logit + std * torch.randn(mean_logit.shape, device=mean_logit.device)).sigmoid()
    #     else:
    #         att_bern = (mean_logit).sigmoid()
    #     return att_bern