import os
import torch
import logging
import random
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from models.hyponet import get_hyponet
from models.hrnet import get_pose_net
from models.scorenet import get_score_net
from models.ema import ExponentialMovingAverage
from models.loss import SMPL_LOSS, SCORE_LOSS
from utils.filter_hub import *

from dataset.mix_dataset import MixDataset
from dataset.h36m import H36MDataset as h36m
from dataset.pw3d import PW3D as pw3d

def _init_fn(worker_id):
    np.random.seed(123)
    random.seed(123)

def get_dataloader(config, is_train = True):
    if is_train:
        datasets = {'mix': MixDataset(cfg=config,train=True)}
    else:
        dataset1 = h36m(
                cfg=config,
                ann_file=config.dataset.set_list[0].test_set,
                root = config.dataset.set_list[0].root,
                train=False)
        dataset2 = pw3d(
                cfg=config,
                ann_file=config.dataset.set_list[3].test_set,
                root = config.dataset.set_list[3].root,
                train=False)
        datasets = {'h36m': dataset1, '3dpw':dataset2}
    dataloaders = {}
    samplers = {}
    for key, dataset in datasets.items():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
        batch_size = config.sampling.batch_size
        if is_train:
            shuffle = (sampler is None)
            batch_size = config.training.batch_size
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=config.dataset.workers,
                sampler=sampler,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=_init_fn,
                )
        dataloaders[key] = dataloader
        samplers[key] = sampler
        logging.info(f"dataset length is {len(dataset)}")
    return dataloaders, datasets, samplers


def get_optimizer(config, parameters,lr):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def get_model(config, is_train = True, resume = False, resume_path = None):
    neighbour_matrix = get_neighbour_matrix_from_hand(parents,childrens,num_joints=config.hyponet.num_joints,num_edges=config.hyponet.num_twists,knn=config.hyponet.knn)
    model = get_hyponet(config, neighbour_matrix, is_train=is_train)
    model = model.to(config.device)
    model_cond = get_pose_net(config,is_train=is_train).to(config.device)

    ema_helper = ExponentialMovingAverage(model.parameters(), decay=config.hyponet.ema_rate)
    ema_helper_cond = ExponentialMovingAverage(model_cond.parameters(), decay=config.hyponet.ema_rate)

    optimizer_hyponet, optimizer_hrnet, loss = None, None, None
    if is_train:
        optimizer_hyponet = get_optimizer(config, model.parameters(),lr=config.optim.lr_model)
        backbone_params = list(map(id, model_cond.incre_modules.parameters())) + \
                list(map(id, model_cond.downsamp_modules.parameters())) + \
                list(map(id, model_cond.final_feat_layer.parameters())) + \
                list(map(id, model_cond.pred_beta.parameters())) + \
                list(map(id, model_cond.fmap_layer.parameters())) + \
                list(map(id, model_cond.hmap_layer.parameters())) + \
                list(map(id, model_cond.fmap_layer_local.parameters()))
        logits_params = filter(lambda p: id(p) not in backbone_params, model_cond.parameters())
        finetune_params = filter(lambda p: id(p) in backbone_params, model_cond.parameters())
        optim_list =[{"params":logits_params,"lr":config.optim.lr_hrnet[0]},
                     {"params":finetune_params,"lr":config.optim.lr_hrnet[1]}]
        optimizer_hrnet = torch.optim.Adam(optim_list)

        loss = SMPL_LOSS(config).to(config.device)
    
    start_epoch, step = 0, 0
    min_mpjpe_pw3d, min_mpjpe_h36m = 1e10, 1e10
    if resume:
        states = torch.load(resume_path,map_location='cpu')
        start_epoch = states['epoch']+1
        step = states['step']
        if 'min_mpjpe_pw3d' in states:
            min_mpjpe_pw3d = states['min_mpjpe_pw3d']
        if 'min_mpjpe_h36m' in states:
            min_mpjpe_h36m = states['min_mpjpe_h36m']
        model.load_state_dict(states['model'])
        model_cond.load_state_dict(states['model_cond'])
        ema_helper.load_state_dict(states['ema'])
        ema_helper.to(config.device)
        ema_helper_cond.load_state_dict(states['ema_cond'])
        ema_helper_cond.to(config.device)
        if is_train:
            optimizer_hrnet.load_state_dict(states['optimizer_cond'])
            optimizer_hyponet.load_state_dict(states['optimizer'])
        print(f"resume from {resume_path}")
        if config.training.finetune:
            for param_group in optimizer_hrnet.param_groups:
                param_group['lr'] /= 2
            for param_group in optimizer_hyponet.param_groups:
                param_group['lr'] /= 2
    if is_train:
        model.train()
        model_cond.train()
    else:
        model.eval()
        model_cond.eval()
        ema_helper.copy_to(model.parameters())
        ema_helper_cond.copy_to(model_cond.parameters())
    return model, model_cond, ema_helper, ema_helper_cond, optimizer_hyponet, optimizer_hrnet, start_epoch, step, loss, min_mpjpe_h36m, min_mpjpe_pw3d

def load_model(model,state):
    state_model = model.state_dict()
    for key in state_model.keys():
        if key in state.keys() and state_model[key].shape == state[key].shape:
            state_model[key] = state[key]
    model.load_state_dict(state_model)
    return model

def get_model_score(config, is_train = True, resume = False, resume_path = None):
    neighbour_matrix = get_neighbour_matrix_from_hand(parents,childrens,num_joints=config.hyponet.num_joints,num_edges=config.hyponet.num_twists,knn=config.scorenet.knn)
    model_score = get_score_net(config,neighbour_matrix, is_train=is_train).to(config.device)
    model_score_cond = get_pose_net(config, is_train=is_train, score=True).to(config.device)

    if config.training.scorenet.load_weight:
        states_load = torch.load(config.training.scorenet.gen_path[-1], map_location='cpu')
        model_score = load_model(model_score, states_load['model'])
        model_score_cond = load_model(model_score_cond, states_load['model_cond'])

    ema_score = ExponentialMovingAverage(model_score.parameters(), decay=config.scorenet.ema_rate)
    ema_score_cond = ExponentialMovingAverage(model_score_cond.parameters(), decay=config.scorenet.ema_rate)
    
    optimizer_score, optimizer_score_cond, loss = None, None, None
    if is_train:
        optimizer_score = get_optimizer(config, model_score.parameters(),lr=config.optim.lr_model)
        backbone_params = list(map(id,model_score_cond.fmap_layer.parameters())) + list(map(id,model_score_cond.hmap_layer.parameters())) + list(map(id,model_score_cond.fmap_layer_local.parameters()))
        logits_params = filter(lambda p: id(p) not in backbone_params, model_score_cond.parameters())
        ft_params = filter(lambda p: id(p) in backbone_params, model_score_cond.parameters())
        optim_list = [{"params": ft_params,"lr":config.optim.lr_hrnet[1]},
                    {"params":logits_params,"lr":config.optim.lr_hrnet[0]}]
        optimizer_score_cond = torch.optim.Adam(optim_list)
        loss = SCORE_LOSS(config).to(config.device)
    
    start_epoch, step = 0, 0
    if resume:
        states = torch.load(resume_path,map_location='cpu')
        start_epoch = states['epoch']+1
        step = states['step']
        model_score.load_state_dict(states['model_score'])
        model_score_cond.load_state_dict(states['model_score_cond'])
        ema_score.load_state_dict(states['ema_score'])
        ema_score.to(config.device)
        ema_score_cond.load_state_dict(states['ema_score_cond'])
        ema_score_cond.to(config.device)
        if is_train:
            optimizer_score.load_state_dict(states['optimizer_score'])
            optimizer_score_cond.load_state_dict(states['optimizer_score_cond'])
        print(f"resume from {resume_path}")
    
    if is_train:
        model_score.train()
        model_score_cond.train()
    else:
        model_score.eval()
        model_score_cond.eval()
        ema_score.copy_to(model_score.parameters())
        ema_score_cond.copy_to(model_score_cond.parameters())
    return model_score, model_score_cond, ema_score, ema_score_cond, optimizer_score, optimizer_score_cond, start_epoch, step, loss

def process_pred(pred,dataset,multi_n,type='H36M',save_path=None,use_score=False):         
    error_dict_all = {'mpjpe': [], 'pa_mpjpe':[],'PVE':[],'score':[]}
    pred0 = {'joint_cam':[],'joint_uvd':[],'twist':[],'shape':[],'pred':[],'theta':[]}
    for midx in tqdm(range(multi_n)):
        error_dict,pred_return = dataset.evaluate_xyz_17(pred[midx],save_path+f"_{type}.json")
        error_dict_all['mpjpe'].append(error_dict['error_all_joint'])
        error_dict_all['pa_mpjpe'].append(error_dict['error_all_joint_gt'])
        error_dict_all['PVE'].append(error_dict['PVE'])
        error_dict_all['score'].append(error_dict['score'])
        for key in pred0.keys():
            pred0[key].append(pred_return[key])
    mpjpe_all = np.mean(np.array(error_dict_all['mpjpe']), -1)
    pa_mpjpe_all = np.mean(np.array(error_dict_all['pa_mpjpe']), -1)
    pve_all = np.array(error_dict_all['PVE'])
    score_all = np.array(error_dict_all['score'])
    min_idx = np.argmax(score_all,axis=0)
    idx = np.arange(0,score_all.shape[1])
    mpjpe_select = mpjpe_all[min_idx,idx].mean()
    pa_mpjpe_select = pa_mpjpe_all[min_idx,idx].mean()
    pve_select = pve_all[min_idx,idx].mean()

    mpjpe = torch.tensor(mpjpe_all).transpose(1,0)
    pa_mpjpe = torch.tensor(pa_mpjpe_all).transpose(1,0)
    pve = torch.tensor(pve_all).transpose(1,0)
    score = torch.tensor(score_all).transpose(1,0)
    mpjpe_min = np.min(mpjpe_all, axis=0).mean()
    pa_mpjpe_min = np.min(pa_mpjpe_all, axis=0).mean()
    pve_min = np.min(pve_all,axis=0).mean()
    if save_path is not None:
        save_dir = os.path.join(save_path,str(multi_n),type)
        os.makedirs(save_dir,exist_ok=True)
        for key in pred0.keys():
            save_path  = os.path.join(save_dir,'{}.npy'.format(key))
            np.save(save_path,np.array(pred0[key]))
        for key in error_dict_all.keys():
            path_dst = os.path.join(save_dir,'{}.npy'.format(key))
            np.save(path_dst,np.array(error_dict_all[key]))
    if not use_score:
        mpjpe_select, pa_mpjpe_select, pve_select = -1, -1, -1
    z0_idx = min(multi_n-1,1-(not use_score))
    logging.info(f"{type}: validate mpjpe:{mpjpe_min:.1f}mm mpjpe_select:{mpjpe_select:.1f}mm  mpjpe_z0:{mpjpe_all[z0_idx].mean():.1f}mm mpjpe_mean:{mpjpe_all[-1].mean():.1f}")
    logging.info(f"{type}: validate pa-mpjpe:{pa_mpjpe_min:.1f}mm pa-mpjpe_select:{pa_mpjpe_select:.1f}mm  pa-mpjpe_z0:{pa_mpjpe_all[z0_idx].mean():.1f}mm pa-mpjpe_mean:{pa_mpjpe_all[-1].mean():.1f}")
    logging.info(f"{type}: validate pve:{pve_min:.1f}mm pve_select:{pve_select:.1f}mm  pve_z0:{pve_all[z0_idx].mean():.1f}mm pve_mean:{pve_all[-1].mean():.1f}")
    if use_score:
        return {'mpjpe':mpjpe_select,'pampjpe':pa_mpjpe_select,'pve':pve_select}
    else:
        return {'mpjpe':mpjpe_all[z0_idx].mean(), 'pampjpe':pa_mpjpe_all[z0_idx].mean(), 'pve':pve_all[z0_idx].mean()}
