import math
import torch
import torch.nn as nn
import numpy as np
from utils.relation import *
import functools

def twist_to_angle(twist):
    select_angle = torch.arctan(twist[:,:,:,1]/twist[:,:,:,0]) 
    flag = (torch.cos(select_angle)*twist[:,:,:,0])<0
    select_angle = select_angle + flag*np.pi
    return select_angle

def weighted_l1_loss(input, target, weights, size_average=True):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()

class SCORE_LOSS(nn.Module): 
    def __init__(self, cfg):
        super(SCORE_LOSS, self).__init__()
        self.eps = cfg.loss.scorenet.eps
        self.weight_pve = cfg.loss.scorenet.weight_pve
        self.weight_mcam = cfg.loss.scorenet.weight_mcam
        self.weight_2d = cfg.loss.scorenet.weight_2d
        self.num_joints =cfg.hyponet.num_joints
        self.num_twists = cfg.hyponet.num_twists
        self.sigma = cfg.loss.scorenet.sigma
        self.loss_func = functools.partial(rankloss,sigma=self.sigma)
    def forward(self,score,output,gt,mask):
        bs = gt['mesh'].shape[0]
        assert output['pred_vertices'].shape[0] % bs==0
        multi_n = output['pred_vertices'].shape[0] // bs
        pred_mesh = output['pred_vertices'].view(bs,multi_n,-1,3)
        gt_mesh = gt['mesh'].unsqueeze(1)
        pve = torch.sqrt(torch.sum((pred_mesh - gt_mesh) ** 2, dim=3))
        pve = torch.mean(pve,dim=2)* 1000
        
        joint_cam_pred = output['pred_xyz_jts_17'].reshape(bs,multi_n,-1,3)
        joint_cam_pred = joint_cam_pred - joint_cam_pred[:,:,0].reshape(bs,multi_n,1,3)
        joint_cam_pred = joint_cam_pred*2000
        joint_cam_gt = gt['joint_cam'] - gt['joint_cam'][:,0].unsqueeze(1)
        joint_cam_gt = joint_cam_gt.unsqueeze(1)
        mpjpe_cam = torch.sqrt(torch.sum((joint_cam_gt - joint_cam_pred) ** 2, dim=3))
        mpjpe_cam = torch.mean(mpjpe_cam,dim=2)
        
        score_t = score.view(bs,multi_n)
        p_pve = self.loss_func(score_t.clone(),pve.clone(),mask.clone())*self.weight_pve
        p_mpjpe_cam = self.loss_func(score_t.clone(),mpjpe_cam.clone(),mask.clone())*self.weight_mcam
        
        
        pred_2d =  output['pred_2d'].view(-1,self.num_joints,2)
        gt_2d = gt['pred_2d'].view(-1,self.num_joints,2)
        loss_2d =  (pred_2d-gt_2d).square().view(-1,self.num_joints,2)*gt['mask_2d']
        loss_2d = loss_2d.view(-1,self.num_joints*2).sum(dim=1).mean(dim=0) * self.weight_2d
        
        return p_pve, p_mpjpe_cam, loss_2d
            

class SMPL_LOSS(nn.Module):
    def __init__(self,cfg):
        super(SMPL_LOSS, self).__init__()
        self.weight_shape = cfg.loss.hyponet.weight_shape
        self.weight_diff = cfg.loss.hyponet.weight_diff
        self.criterion_smpl = nn.MSELoss()
        self.weight_2d = cfg.loss.hyponet.weight_2d
        self.num_joints = cfg.hyponet.num_joints

    def forward(self,output, gt, labels):
        loss_beta = (output['pred_shape']-labels['target_beta'])*labels['target_smpl_weight']
        loss = loss_beta.square().sum(dim=1).mean(dim=0)*self.weight_shape

        loss_2d = (output['joint_2d']-gt['joint_2d']).square().view(-1,self.num_joints,2)*labels['joints_vis_29'][:,:,:2]
        loss_2d = loss_2d.sum(dim=-1).sum(dim=-1).mean()*self.weight_2d
        loss += loss_2d
       
        loss_joint = (output['noise_j']-gt['noise_j']).square().view(-1,self.num_joints,3)
        loss_joint = loss_joint *labels['joints_vis_29']
        loss_joint = loss_joint.view(-1,self.num_joints*3).sum(dim=1).mean(dim=0)*self.weight_diff
        loss_twist = (output['noise_t']-gt['noise_t']).square().view(-1,23,2)
        loss_twist = loss_twist*labels['target_twist_weight']
        loss_twist = loss_twist.view(-1,23*2).sum(dim=1).mean(dim=0)*self.weight_diff
        loss += (loss_twist+loss_joint)

        assert torch.isnan(loss).sum()==0
        return loss,{'loss_2d':loss_2d,'loss_twist':loss_twist,'loss_joint':loss_joint,'loss_beta':loss-loss_joint-loss_twist-loss_2d}
