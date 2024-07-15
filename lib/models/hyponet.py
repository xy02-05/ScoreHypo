import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
import pickle, h5py
import logging
import math

logger = logging.getLogger(__name__)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    from https://github.com/wzlxjtu/PositionalEncoding2D
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias!=None:
            m.bias.data.fill_(0.01)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def clip_by_norm(layer, norm=1):
    if isinstance(layer, nn.Linear):
        if layer.weight.data.norm(2) > norm:
            layer.weight.data.mul_(norm / layer.weight.data.norm(2).item())

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super(DecoderLayer,self).__init__()
        self.local_ch = cfg.hrnet.local_ch
        self.joint_ch = cfg.hyponet.joint_ch + self.local_ch
        self.dropout_rate = 0.1
        self.joints = cfg.hyponet.num_joints
        self.edges = cfg.hyponet.num_twists
        self.num_item = self.edges+self.joints
        feedforward_dim = self.joint_ch*4

        self.norm1 = nn.LayerNorm(self.joint_ch)
        self.self_attn = nn.MultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.hyponet.heads,batch_first = True)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        self.norm2 = nn.LayerNorm(self.joint_ch)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.joint_ch, num_heads=cfg.hyponet.heads,batch_first = True)
        self.dropout2 = nn.Dropout(p=self.dropout_rate) 

        self.linear1 = nn.Linear(self.joint_ch, feedforward_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.linear2 = nn.Linear(feedforward_dim, self.joint_ch)
        self.norm3 = nn.LayerNorm(self.joint_ch)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)
        self.activation = nn.ReLU()
    def with_pos_embed(self, tensor, pos):
        return tensor + pos
    
    def forward(self, tgt, memory,mask= None,mask_ctx = None,pos= None,pos_ctx=None, gen_multi=False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        if gen_multi:
            bs = memory.shape[0]
            multi_n = tgt.shape[0] // bs
            pos = pos.repeat(1,multi_n,1).view(-1,self.num_item*multi_n,self.joint_ch)
            tgt2 = self.norm2(tgt).view(bs,-1,self.joint_ch)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos),
                                    key=self.with_pos_embed(memory, pos_ctx),
                                    value=memory, attn_mask=None)[0].contiguous().view(bs*multi_n,-1,self.joint_ch)
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos),
                                    key=self.with_pos_embed(memory, pos_ctx),
                                    value=memory, attn_mask=None)[0]
            tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class HypoNet(nn.Module):
    def __init__(self, cfg, neighbour_matrix=None):
        super(HypoNet, self).__init__()
        self.joint_ch = cfg.hyponet.joint_ch
        self.local_ch = cfg.hrnet.local_ch
        self.ch = cfg.hyponet.joint_ch + self.local_ch
        self.joints = cfg.hyponet.num_joints
        self.edges = cfg.hyponet.num_twists
        self.num_item = self.edges+self.joints
        self.num_blocks = cfg.hyponet.num_blocks
        self.dropout_rate = 0.25
        # mask_matrix
        self.neighbour_matrix = neighbour_matrix
        self.mask = self.init_mask(self.neighbour_matrix[0])
        self.mask_twist = self.init_mask(self.neighbour_matrix[1])
        self.mask_joints = self.init_mask(self.neighbour_matrix[2])
        self.mask_list = []
        self.atten_knn = cfg.hyponet.atten_knn
        assert len(self.atten_knn) == self.num_blocks
        for i in range(self.num_blocks):
            mask_i = np.linalg.matrix_power(neighbour_matrix[3], self.atten_knn[i])
            mask_i = np.array(mask_i!=0, dtype=np.float32)
            mask_i = self.init_mask(mask_i)
            mask_i = 1 - mask_i
            self.mask_list.append(mask_i.bool())

        # first layer
        self.linear_start_j = nn.Linear(self.joints*3, self.joints*self.joint_ch)
        self.bn_start_j = nn.GroupNorm(32, num_channels=self.joints*self.joint_ch)
        self.activation_start_j = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_start_j = nn.Dropout(p=self.dropout_rate)
        # twist
        self.linear_start_t = nn.Linear(self.edges*2, self.edges*self.joint_ch)
        self.bn_start_t = nn.GroupNorm(32, num_channels=self.edges*self.joint_ch)
        self.activation_start_t = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_start_t = nn.Dropout(p=self.dropout_rate)

        self.emb_h = self.emb_w = 8

        # final layer
        self.linear_final_j = nn.Linear(self.joints*self.ch, self.joints*3)
        self.linear_final_t = nn.Linear(self.edges*self.ch, self.edges*2)
         
        #blocks
        self.blocks = nn.ModuleList([DecoderLayer(cfg) for i in range(self.num_blocks)])

        # time
        self.temb_ch =cfg.hyponet.temb_ch 
        self.temb_dense = nn.ModuleList([torch.nn.Linear(self.temb_ch,self.temb_ch*4),
                                        torch.nn.Linear(self.temb_ch*4,self.ch),])

    def init_mask(self,neighbour_matrix):
        """
        Only support locally_connected
        """
        L = neighbour_matrix.T
        return torch.from_numpy(L)

    def mask_weights(self, layer,mask,mshape):
        assert isinstance(layer, nn.Linear), 'masked layer must be linear layer'

        output_size, input_size = layer.weight.shape  # pytorch weights [output_channel, input_channel]
        input_size, output_size = int(input_size), int(output_size)
        assert input_size % mshape == 0 and output_size % mshape == 0
        in_F = int(input_size / mshape)
        out_F = int(output_size / mshape)
        weights = layer.weight.data.view([mshape, out_F, mshape, in_F])
        weights.mul_(mask.t().view(mshape, 1, mshape, 1).to(device=weights.get_device()))
    
    def forward(self, xinj,xint,t,ctx=None, gen_multi=False):
        bs = ctx['global'].shape[0]
        if gen_multi:
            assert xinj.shape[0]%bs==0
            multi_n = xinj.shape[0] // bs
        self.mask_weights(self.linear_start_j,self.mask_joints,self.joints)
        self.mask_weights(self.linear_start_t,self.mask_twist,self.edges)
        self.mask_weights(self.linear_final_j,self.mask_joints,self.joints)
        self.mask_weights(self.linear_final_t,self.mask_twist,self.edges)

        # condition 
        emb_ctx = positionalencoding2d(self.ch, self.emb_h, self.emb_w).unsqueeze(dim=0).cuda()
        emb_ctx = torch.flatten(emb_ctx,start_dim=2, end_dim=3).transpose(2,1)
        ctx_global = torch.flatten(ctx['global'],start_dim=2, end_dim=3).transpose(2,1)
        ctx_local = ctx['local'].view(-1,self.num_item,self.local_ch)
        if gen_multi:
            ctx_local = ctx_local.unsqueeze(1).repeat(1,multi_n,1,1).view(-1,self.num_item,self.local_ch)
        emb = get_timestep_embedding(torch.arange(0,self.num_item),self.ch).cuda().view(1,self.num_item,self.ch).cuda()

        # time
        temb = get_timestep_embedding(t, self.temb_ch)
        temb = self.temb_dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense[1](temb)
        temb = nonlinearity(temb)
        temb = temb.unsqueeze(1).repeat(1,self.num_item,1)
        temb = temb.view(-1,self.num_item,self.ch)

        # first layer
        # joints
        x0 = self.linear_start_j(xinj)
        x0 = self.bn_start_j(x0)
        x0 = self.activation_start_j(x0)
        x0 = self.dropout_start_j(x0)
        x0 = x0.contiguous().view(-1, self.joints,self.joint_ch)
        # twist
        x1 = self.linear_start_t(xint)
        x1 = self.bn_start_t(x1)
        x1 = self.activation_start_t(x1)
        x1 = self.dropout_start_t(x1)
        x1 =  x1.contiguous().view(-1, self.edges,self.joint_ch)

        x = torch.cat([x0,x1],dim=1).view(-1,self.num_item,self.joint_ch)
        x = torch.cat([x,ctx_local],dim=-1).view(-1,self.num_item,self.ch)
        x += temb

        for block_idx in range(self.num_blocks):
            x = self.blocks[block_idx](tgt=x, memory=ctx_global,pos = emb,pos_ctx = emb_ctx,mask = self.mask_list[block_idx].clone().bool().to(x.device),gen_multi=gen_multi)

        x = x.view(-1,self.num_item,self.ch)
        xj = x[:,:self.joints,:].view(-1,self.joints*self.ch)
        xt = x[:,self.joints:,:].view(-1,self.edges*self.ch)
        xj = self.linear_final_j(xj) 
        xj = xj + xinj
        xt = self.linear_final_t(xt) 
        xt = xt + xint
        return xj,xt

def get_hyponet(cfg, neighbour_matrix, is_train):
    model = HypoNet(cfg,neighbour_matrix)
    pretrained = cfg.hyponet.pretrained

    if is_train:
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained lcn model {}'.format(pretrained))
            model.load_state_dict(pretrained_state_dict['state_dict_3d'], strict=True)
        else:
            logger.info('=> init lcn weights from kaiming normal distribution')
            model.apply(init_weights)
            model.apply(clip_by_norm)
    return model
