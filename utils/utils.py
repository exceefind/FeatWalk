
import os
import shutil
import time
import pprint
import torch
import numpy as np
import os.path as osp
import random
import torch.nn.functional as F

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model, dir):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    for k, v in file_dict.items():
        if k not in model_dict:
            print(k)
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model

def compute_weight_local(feat_g,feat_ql,feat_sl,temperature=2.0):
    # feat_g : nk * dim
    # feat_l : nk * m * dim
    [_,k,m,dim] = feat_sl.shape
    [n,q,m,dim] = feat_ql.shape
    # print(feat_ql.shape)

    # 先计算除局部i外的局部与自身之间的相似度之和:
    # 表征互补性
    feat_g_expand = feat_g.unsqueeze(2).expand_as(feat_ql)
    # print(feat_g_expand.shape)
    # print(feat_l.shape)
    sim_gl = torch.cosine_similarity(feat_g_expand,feat_ql,dim=-1)
    # print(sim_gl.shape)
    I_opp_m = (1 - torch.eye(m)).unsqueeze(0).to(sim_gl.device)
    # I_opp_m = torch.ones((m,m)).unsqueeze(0).to(sim_gl.device)
    # I_opp_m = torch.eye(m).unsqueeze(0).to(sim_gl.device)
    # print(I_opp_m.shape)
    # sim_gl : n * k *  1 * m
    sim_gl = -(torch.matmul(sim_gl, I_opp_m).unsqueeze(-2))/(m-1)
    # sim_gl = (torch.matmul(sim_gl, I_opp_m).unsqueeze(-2))/m
    # sim_gl = (torch.matmul(sim_gl, I_opp_m).unsqueeze(-2))
    # print(sim_gl.shape)

    # # 计算局部i与其他类局部的平均匹配度:
    # # 作为分母，表征类相关性
    # feat_sl_flatten = feat_sl.reshape(-1,dim)
    # feat_ql_flatten = feat_ql.reshape(-1,dim)
    # sim_ll = torch.cosine_similarity(feat_ql_flatten.unsqueeze(0),feat_sl_flatten.unsqueeze(1),dim=-1).view(n,q,m,n,-1)
    # sim_ll = torch.mean(sim_ll,dim=-1)
    # I_opp_n = (1 - torch.eye(n)).unsqueeze(0).unsqueeze(1).to(sim_ll.device)
    # sim_ll = (torch.matmul(sim_ll,I_opp_n).transpose(-1,-2)) / (n-1)

    # print(sim_ll.shape)
    # print(sim_gl.shape)
    # wight = F.softmax((sim_gl / sim_ll)/temperature,dim=-1)

    return sim_gl

#  proto_walk
def compute_weight_local(feat_g,feat_ql,feat_sl,measure = "cosine"):
    # feat_g : nk * dim
    # feat_l : nk * m * dim
    [_,k,m,dim] = feat_sl.shape
    [n,q,m,dim] = feat_ql.shape
    # print(feat_ql.shape)

    # 先计算除局部i外的局部与自身之间的相似度之和:
    feat_g_expand = torch.mean(feat_g,dim=1).unsqueeze(0).unsqueeze(1).unsqueeze(3)
    # print(feat_g_expand.shape)
    # print(feat_l.shape)
    if measure == "cosine":
        sim_gl = torch.cosine_similarity(feat_g_expand,feat_ql.unsqueeze(2),dim=-1)
    else:
        # print("eudist")
        sim_gl = -1 * 0.001 * torch.sum((feat_g_expand - feat_ql.unsqueeze(2)) ** 2, dim=-1)
        sim_gl = -1 * 0.002 * torch.sum((feat_g_expand - feat_ql.unsqueeze(2)) ** 2, dim=-1)

    # print(sim_gl.shape)
    # I_opp_m = (1 - torch.eye(m)).unsqueeze(0).unsqueeze(1).to(sim_gl.device)
    I_m = torch.eye(m).unsqueeze(0).unsqueeze(1).to(sim_gl.device)
    # I_opp_m = torch.ones((m,m)).unsqueeze(0).to(sim_gl.device)
    # I_opp_m = torch.eye(m).unsqueeze(0).to(sim_gl.device)
    # print(I_opp_m.shape)
    # sim_gl : n * k *  1 * m
    # sim_gl = -200 * (torch.matmul(sim_gl, I_opp_m))/(m-1)
    sim_gl =  torch.matmul(sim_gl, I_m)

    # print(sim_gl.shape)
    return sim_gl

# # eudist
# def compute_weight_local(feat_g,feat_ql,feat_sl,temperature=2.0):
#     # feat_g : nk * dim
#     # feat_l : nk * m * dim
#     [_,k,m,dim] = feat_sl.shape
#     [n,q,m,dim] = feat_ql.shape
#     # print(feat_ql.shape)
#
#     # 先计算除局部i外的局部与自身之间的相似度之和:
#     feat_g_expand = torch.mean(feat_g,dim=1).unsqueeze(0).unsqueeze(1).unsqueeze(3)
#     # print(feat_g_expand.shape)
#     # print(feat_l.shape)
#     sim_gl = -torch.sum((feat_g_expand-feat_ql.unsqueeze(2))**2,dim=-1)
#     # print(sim_gl.shape)
#     I_m = torch.eye(m).unsqueeze(0).unsqueeze(1).to(sim_gl.device)
#     # I_opp_m = torch.ones((m,m)).unsqueeze(0).to(sim_gl.device)
#     # I_opp_m = torch.eye(m).unsqueeze(0).to(sim_gl.device)
#     # print(I_opp_m.shape)
#     # sim_gl : n * k *  1 * m
#     sim_gl = -0.002 * (torch.matmul(sim_gl, I_m))
#
#     # print(sim_gl.shape)
#     return sim_gl



if __name__ == '__main__':
    feat_g = torch.randn((5,15,64))
    # feat_g = torch.ones((5,3,64))
    feat_sl = torch.randn((5,3,6,64))
    feat_ql = torch.randn((5,15,6,64))
    # feat_l = torch.ones((5,3,6,64))
    compute_weight_local(feat_g,feat_ql,feat_sl)
    # print(compute_weight_local(feat_g,feat_ql,feat_sl)[0,0])