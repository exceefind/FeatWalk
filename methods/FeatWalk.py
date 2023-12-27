import os
import sys
import time

import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from methods.bdc_module import BDC
import torch.nn.functional as F

sys.path.append("..")
import numpy as np
import torch.nn as nn
import torch
import scipy
from scipy.stats import t
import network.resnet as resnet
# from .DN4_module import *
import random
from utils.loss import *
from sklearn.linear_model import LogisticRegression as LR
# torch.autograd.set_detect_anomaly(True)
from utils.distillation_utils import *
from utils.utils import *
import math
from torch.nn.utils.weight_norm import WeightNorm

import warnings
warnings.filterwarnings("ignore")

train_loss_list = []
test_loss_list = []
prototypes_changes_list = []


def mean_confidence_interval(data, confidence=0.95,multi = 1):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m * multi, h * multi

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def random_sample(linspace, max_idx, num_sample=5):
    sample_idx = np.random.choice(range(linspace), num_sample)
    sample_idx += np.sort(random.sample(list(range(0, max_idx, linspace)),num_sample))
    return sample_idx

def Triuvec(x,no_diag = False):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    # I = torch.eye(dim,dim).reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y

def Triumap(x,no_diag = False):

    batchSize, dim, dim, h, w = x.shape
    r = x.reshape(batchSize, dim * dim, h, w)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    # I = torch.eye(dim,dim).reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index, :, :].squeeze()
    return y

def Diagvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.eye(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = r[:, index].squeeze()
    return y

class reconstruct_layer(nn.Module):
    def __init__(self, in_channels = 128, out_channels=128, p_des=0.2, p_drop = 0.5,skip_connect=True):
        super(reconstruct_layer, self).__init__()
        # init : p_des = 0.2 p_drop=0.6
        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1, bias=False)
        # self.conv_soft = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=1, padding=1, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv_soft = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,  bias=False)
        self.conv_rec = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

        )
        self.skip_connect = skip_connect
        self.drop_des2 = nn.Dropout2d(p_des)
        self.drop_des = nn.Dropout(p_des)
        self.p_drop = p_drop
        self.scale = torch.nn.Parameter(torch.ones(in_channels),requires_grad=True)
    #     初始化权重：
    #     print(self.conv_soft.weight.data.shape)
    #     self.conv_soft.weight.data[:,:,0,0] = torch.eye(in_channels)

    def forward(self, x):
        mask = torch.ones((x.shape[0], x.shape[1])).to(x.device)
        if random.random() <= self.p_drop:
            # map_mask = (self.drop_des(mask)/(mask)).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            map_mask = (self.drop_des(mask)).unsqueeze(-1).unsqueeze(-1).expand_as(x)

        else:
            map_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        # print(self.conv.weight.shape)
        # map_mask = self.drop_des2(mask)
        # print(self.conv_soft.weight.data)
        # print(self.conv_soft.weight.data.shape)
        # w_soft = F.softmax(self.conv.weight.data,dim=1)
        w_soft = F.sigmoid(self.conv_soft.weight.data)
        # w_soft = self.conv_soft.weight.data
        # w_soft = w_soft.clamp(0,1)

        # w_soft = self.conv_soft.weight.data
        # w_soft_nrom = torch.norm(w_soft, p=2, dim=1).unsqueeze(1).expand_as(w_soft)
        # w_soft = w_soft.div(w_soft_nrom + 1e-9)

        self.conv_soft.weight.data = w_soft

        # self.conv_soft.weight.data = F.sigmoid(self.conv.weight.data,)
        # self.conv_soft.weight.data.clamp_(-.5,.5)
        # x_mask = x*map_mask
        # x_mask = self.drop_des2(x)
        out = self.conv_soft(x)
        # out = self.conv_rec(x)
        # return out
        if self.skip_connect:
            # self.scale.data.clamp_(0.5,2)
            out = torch.cat([out.unsqueeze(-1), x.unsqueeze(-1)],dim=-1)
            # return out
            return torch.mean(out,dim=-1)
            # return x

        else:
            return out

class fusion_module(nn.Module):
    def __init__(self,in_channels=2,out_channels=1,dim=8256):
        super(fusion_module, self).__init__()
        self.conv =  nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv_soft =  nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.fusion = nn.Sequential(
            # nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),

        )
        self.weight =  torch.nn.Parameter(torch.randn(dim),requires_grad=True)

    def _init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight, a=0, b=1)
                # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')


    def forward(self,x):
        # print(self.conv_soft.weight.data.shape)
        # w_soft = F.softmax(self.conv.weight.data,dim=1)
        # self.conv_soft.weight.data = w_soft
        # x = self.conv_soft(x)
        # print(x.shape)
        weight = self.weight.unsqueeze(0).expand(x.shape[0],x.shape[-1])
        x = F.sigmoid(weight) * x[:,0,:] + (1-F.sigmoid(weight)) * x[:,1,:]

        return self.fusion(x)


class FeatWalk_Net(nn.Module):
    def __init__(self,params,num_classes = 5,):
        super(FeatWalk_Net, self).__init__()

        self.params = params
        self.out_map = False

        if params.model == 'resnet12':
            self.backbone = resnet.ResNet12(avg_pool=True,num_classes=64)
            resnet_layer_dim = [64, 160, 320, 640]
        elif params.model == 'resnet18':
            self.backbone = resnet.ResNet18()
            resnet_layer_dim = [64, 128, 256, 512]

        self.resnet_layer_dim = resnet_layer_dim
        self.reduce_dim = params.reduce_dim
        self.feat_dim = self.backbone.feat_dim
        self.dim = int(self.reduce_dim * (self.reduce_dim+1)/2)
        if resnet_layer_dim[-1] != self.reduce_dim:
            self.Conv = nn.Sequential(
                nn.Conv2d(resnet_layer_dim[-1], self.reduce_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.reduce_dim),
                nn.ReLU(inplace=True)
            )
            self._init_weight(self.Conv.modules())

        drop_rate = params.drop_rate
        if self.params.embeding_way in ['BDC']:
            self.SFC = nn.Linear(self.dim, num_classes)
            self.SFC.bias.data.fill_(0)
        elif self.params.embeding_way in ['baseline++']:
            self.SFC = nn.Linear(self.reduce_dim, num_classes, bias=False)
            WeightNorm.apply(self.SFC, 'weight', dim=0)
        else:
            self.SFC = nn.Linear(self.reduce_dim, num_classes)

        self.drop = nn.Dropout(drop_rate)

        self.temperature = nn.Parameter(torch.log((1. /(2 * self.feat_dim[1] * self.feat_dim[2])* torch.ones(1, 1))),
                                            requires_grad=True)

        self.dcov = BDC(is_vec=True, input_dim=[self.reduce_dim,self.backbone.feat_dim[1],self.backbone.feat_dim[2]], dimension_reduction=self.reduce_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not self.params.my_model:
            self.feature = self.backbone
        if params.distill:
            if params.embeding_way not in ['GE']:
                self.feature = self.backbone
            if resnet_layer_dim[-1] != self.reduce_dim:
                self.dcov.conv_dr_block = self.Conv
        if not self.params.my_model or self.params.all_mini :
            if params.embeding_way not in ['GE']:
                self.feature = self.backbone
            if resnet_layer_dim[-1] != self.reduce_dim:
                self.dcov.conv_dr_block = self.Conv

        self.n_shot = params.n_shot
        self.n_way = params.n_way
        self.transform_aug = params.n_aug_support_samples

    def _init_weight(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def normalize(self,x):
        x = (x - torch.mean(x, dim=1).unsqueeze(1))
        return x


    def forward_feature(self, x):
        feat_map = self.backbone(x, )
        if self.resnet_layer_dim[-1] != self.reduce_dim:
            feat_map = self.Conv(feat_map)
        out = feat_map
        return out

    def normalize_feature(self, x):
        if self.params.norm == 'center':
            x = x - x.mean(2).unsqueeze(2)
            return x
        else:
            return x

    def forward_feat(self,x,x_c):
        feat_map = self.backbone(x, is_FPN=(self.params.FPN_list is not None))
        feat_map_cl = self.backbone(x_c, is_FPN=(self.params.FPN_list is not None))

        feat_map_1 = self.Conv(feat_map)
        feat_map_2 = self.Conv(feat_map_cl)

        map_mask = torch.ones((feat_map.shape[0], feat_map.shape[1])).cuda()
        # swap_idx = self.drop_swap(map_mask)/(map_mask)
        swap_idx = map_mask / map_mask
        swap_idx = swap_idx.unsqueeze(-1).unsqueeze(-1).expand_as(feat_map)
        # swap_idx = feat_map/(feat_map + 1e-6)
        feat_map_1 = feat_map_1 * swap_idx + feat_map_2 * (1 - swap_idx)
        feat_map_2 = feat_map_1 * (1 - swap_idx) + feat_map_2 * swap_idx

        out_confusion = self.avg_pool(feat_map).view(feat_map.shape[0], -1)
        BDC_1 = self.dcov(feat_map_1)
        BDC_2 = self.dcov(feat_map_2)


        out_1 = self.SFC(self.drop(BDC_1))
        out_2 = self.SFC(self.drop(BDC_2))


        return BDC_1, BDC_2, out_1, out_2

    def forward_pretrain(self, x):
        x = self.forward_feature(x)
        x = self.drop(x)
        return self.SFC(x)

    def train_loop(self,epoch,train_loader,optimizers):
        print_step = 100
        avg_loss = 0
        [optimizer , optimizer_ad] = optimizers
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        loss_ce_fn = nn.CrossEntropyLoss()
        for i ,data in enumerate(train_loader):
            image , label = data
            image = image.cuda()
            label = label.cuda()
            out = self.forward_pretrain(image)
            loss =  loss_ce_fn(out, label)
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total += label.size(0)
            if i % print_step == 0:
                print('\rEpoch {:d} | Batch: {:d}/{:d} | Loss: {:.4f} | Acc_train: {:.2f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1),correct/label.shape[0]*100), end=' ')
        print()

        return avg_loss / iter_num, float(total_correct) / total * 100

    def meta_val_loop(self,epoch,val_loader):
        acc = []
        for i, data in enumerate(val_loader):

            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            split_size = 128
            if support_xs.squeeze(0).shape[0] >= split_size:
                feat_sup_ = []
                for j in range(math.ceil(support_xs.squeeze(0).shape[0] / split_size)):
                    fest_sup_item = self.forward_feature(
                        support_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, support_xs.shape[1]), :, :, :],
                        out_map=self.out_map)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape) >= 1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_, dim=0)
            else:
                feat_sup = self.forward_feature(support_xs.squeeze(0), out_map=self.out_map)
            if query_xs.squeeze(0).shape[0] > split_size:
                feat_qry_ = []
                for j in range(math.ceil(query_xs.squeeze(0).shape[0] / split_size)):
                    feat_qry_item = self.forward_feature(
                        query_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, query_xs.shape[1]), :, :, :],
                        out_map=self.out_map)
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))

                feat_qry = torch.cat(feat_qry_, dim=0)
            else:
                feat_qry = self.forward_feature(query_xs.squeeze(0), out_map=self.out_map)
            if self.params.LR:
                pred = self.LR(feat_sup, support_ys, feat_qry, query_ys)
            else:
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_ys, feat_qry, )
                    _, pred = torch.max(pred, dim=-1)
            if self.params.n_symmetry_aug > 1:
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                query_ys = torch.mode(query_ys, dim=-1)[0]
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            acc.append(acc_epo)
        return mean_confidence_interval(acc)

    def meta_test_loop(self,test_loader):
        acc = []
        for i, (x, _) in enumerate(test_loader):
            self.params.n_aug_support_samples = self.transform_aug
            tic = time.time()
            x = x.contiguous().view(self.n_way, (self.n_shot + self.params.n_queries), *x.size()[2:])
            support_xs = x[:, :self.n_shot].contiguous().view(
                self.n_way * self.n_shot * self.params.n_aug_support_samples, *x.size()[3:]).cuda()
            query_xs = x[:, self.n_shot:, 0:self.params.n_symmetry_aug].contiguous().view(
                self.n_way * self.params.n_queries * self.params.n_symmetry_aug, *x.size()[3:]).cuda()

            support_y = torch.from_numpy(np.repeat(range(self.params.n_way),self.n_shot*self.params.n_aug_support_samples)).unsqueeze(0)
            split_size = 128
            if support_xs.shape[0] >= split_size:
                feat_sup_ = []
                for j in range(math.ceil(support_xs.shape[0]/split_size)):
                    fest_sup_item =self.forward_feature(support_xs[j*split_size:min((j+1)*split_size,support_xs.shape[0]),],out_map=self.out_map)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape)>=1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_,dim=0)
            else:
                feat_sup = self.forward_feature(support_xs,out_map=self.out_map)
            if query_xs.shape[0] >= split_size:
                feat_qry_ = []
                for j in range(math.ceil(query_xs.shape[0]/split_size)):
                    feat_qry_item = self.forward_feature(
                        query_xs[j * split_size:min((j + 1) * split_size, query_xs.shape[0]), ],out_map=self.out_map)
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))

                feat_qry = torch.cat(feat_qry_,dim=0)
            else:
                feat_qry = self.forward_feature(query_xs,out_map=self.out_map)

            if self.params.LR:
                pred = self.predict_wo_fc(feat_sup, support_y, feat_qry,)

            else:
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_y, feat_qry,)
                    if not self.params.LR_rec:
                        _,pred = torch.max(pred,dim=-1)

            query_ys = np.repeat(range(self.n_way), self.params.n_queries)
            pred = pred.view(-1)
            acc_epo = np.mean(pred.cpu().numpy() == query_ys)
            acc.append(acc_epo)
            print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
                                                                                                 *mean_confidence_interval(
                                                                                                     acc, multi=100), (
                                                                                                             time.time() - tic) / 60),
                  end='')

        return mean_confidence_interval(acc)

    def distillation(self,epoch,train_loader,optimizer,model_t):
        print_step = 100
        avg_loss = 0
        total_correct = 0
        iter_num = len(train_loader)
        total = 0
        loss_div_fn = DistillKL(4)
        loss_ce_fn = nn.CrossEntropyLoss()
        # model_t.eval()
        for i, data in enumerate(train_loader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                out_t = model_t.forward_pretrain(image)

            out= self.forward_pretrain(image)
            loss_ce = loss_ce_fn(out, label)
            loss_div = loss_div_fn(out, out_t)

            loss  = loss_ce * 0.5 + loss_div * 0.5
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out, 1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total += label.size(0)
            if i % print_step == 0:
                print('\rEpoch {:d} | Batch: {:d}/{:d} | Loss: {:.4f} | Acc_train: {:.2f}'.format(epoch, i,
                                                                                                  len(train_loader),
                                                                                                  avg_loss / float(
                                                                                                      i + 1),
                                                                                                  correct / label.shape[
                                                                                                      0] * 100),
                      end=' ')
        print()
        return avg_loss / iter_num, float(total_correct) / total * 100

    # new selective local fusion :
    def softmax(self,support_z,support_ys,query_z,):
        # print("--110-")
        loss_ce_fn = nn.CrossEntropyLoss()
        batch_size = self.params.sfc_bs
        walk_times = 24
        alpha = self.params.alpha
        tempe = self.params.sim_temperature


        support_ys = support_ys.cuda()

        if self.params.embeding_way in ['BDC']:
            SFC = nn.Linear(self.dim, self.params.n_way).cuda()

            if self.params.optim in ['Adam']:
                iter_num = 100
                optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                                             weight_decay=self.params.wd_test,eps=1e-4)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_num * math.ceil(self.n_way*self.n_shot/batch_size),eta_min=1e-3)

            else:
                # best
                optimizer = torch.optim.SGD([{'params': SFC.parameters()},{'params':tempe,'lr':0.0}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=self.params.wd_test)
                # 1shot : 69.20+-    5shot 85.73
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.1)
                iter_num = 180
        else:
            tempe =16

            if self.params.embeding_way in ['baseline++']:
                SFC = nn.Linear(self.reduce_dim, self.params.n_way, bias=False).cuda()
                WeightNorm.apply(SFC, 'weight', dim=0)
            else:
                SFC = nn.Linear(self.reduce_dim, self.params.n_way).cuda()

            if self.params.optim in ['Adam']:
                # lr = 5e-3
                optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.005,
                                              weight_decay=self.params.wd_test, eps=5e-3)

                iter_num = 100
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_num * math.ceil(
                    self.n_way * self.n_shot / batch_size), eta_min=5e-3)


            else:
                optimizer = torch.optim.SGD([{'params': SFC.parameters()}],
                                            lr=self.params.lr, momentum=0.9, nesterov=True,
                                            weight_decay=self.params.wd_test)

                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
                iter_num = 180


        SFC.train()

        if self.params.embeding_way in ['BDC']:
            support_z = self.dcov(support_z)
            query_z = self.dcov(query_z)

        else:
            support_z = self.avg_pool(support_z).view(support_z.shape[0], -1)
            query_z = self.avg_pool(query_z)

        support_ys = support_ys.view(self.n_way * self.n_shot, self.params.n_aug_support_samples, -1)
        global_ys = support_ys[:, 0, :]

        support_z = support_z.reshape(self.n_way,self.n_shot,self.params.n_aug_support_samples,-1)
        query_z = query_z.reshape(self.n_way,self.params.n_queries,self.params.n_aug_support_samples,-1)


        feat_q = query_z[:,:,0]
        feat_ql = query_z[:,:,1:]
        feat_g = support_z[:,:,0]
        feat_sl = support_z[:,:,1:]
        # w_local: n * k * n * m
        num_sample = self.n_way*self.n_shot
        global_ys = global_ys.view(self.n_way,self.n_shot,-1)

        feat_g = feat_g.detach()
        feat_sl = feat_sl.detach()

        # feat_sl: n * k * n *  dim

        I = torch.eye(self.n_way,self.n_way,device=feat_g.device).unsqueeze(0).unsqueeze(1)
        proto_moving = torch.mean(feat_g, dim=1)



        for i in range(iter_num):
            weight = compute_weight_local(proto_moving.unsqueeze(1), feat_sl, feat_sl, self.params.measure)
            idx_walk = torch.randperm(self.params.n_aug_support_samples-1,)[:walk_times]
            w_local = F.softmax(weight[:,:,:,idx_walk] * tempe, dim=-1)
            feat_s = torch.sum((feat_sl[:,:,idx_walk,:].unsqueeze(-3)) * (w_local.detach().unsqueeze(-1)), dim=-2)
            support_x = alpha * feat_g.unsqueeze(-2) + (1- alpha) * feat_s
            proto_update = torch.sum(torch.matmul(torch.mean(support_x,dim=1).transpose(1,2),torch.eye(self.n_way,device=proto_moving.device).unsqueeze(0)),dim=-1)
            proto_moving = 0.9 * proto_moving + 0.1 * proto_update
            spt_norm = torch.norm(support_x, p=2, dim=-1).unsqueeze(-1).expand_as(support_x)
            support_x = support_x.div(spt_norm + 1e-6)


        for i in range(iter_num):
            SFC.train()
            sample_idxs = torch.randperm(num_sample)
            for j in range(math.ceil(num_sample/batch_size)):
                idxs = sample_idxs[j*batch_size:min((j+1)*batch_size,num_sample)]
                x =  support_x[idxs//self.n_shot,idxs%self.n_shot]
                y = global_ys[idxs//self.n_shot,idxs%self.n_shot]
                x = self.drop(x)
                out = torch.sum(SFC(x)*I,dim=-1).view(-1,self.n_way)
                loss_ce = loss_ce_fn(out,y.long().view(-1))
                loss = loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

            SFC.eval()

        w_local = compute_weight_local(proto_moving.unsqueeze(1), feat_ql, feat_sl,self.params.measure)
        w_local = F.softmax(w_local *  tempe, dim=-1)

        # feat_sl: n * k * n *  dim
        feat_lq = torch.sum(feat_ql.unsqueeze(-3) * w_local.unsqueeze(-1), dim=-2)
        query_x =  alpha * feat_q.unsqueeze(-2) + (1- alpha) * feat_lq

        spt_norm = torch.norm(query_x, p=2, dim=-1).unsqueeze(-1).expand_as(query_x)
        query_x = query_x.div(spt_norm + 1e-6)

        with torch.no_grad():

            out = torch.sum(SFC(query_x)*I,dim=-1).view(-1,self.n_way)

        return out

    def LR(self,support_z,support_ys,query_z,query_ys):

        clf = LR(penalty='l2',
                 random_state=0,
                 C=self.params.penalty_c,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='multinomial')

        spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        spt_normalized = support_z.div(spt_norm  + 1e-6)

        qry_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        qry_normalized = query_z.div(qry_norm + 1e-6)

        z_support = spt_normalized.detach().cpu().numpy()
        z_query = qry_normalized.detach().cpu().numpy()

        y_support = np.repeat(range(self.params.n_way), self.n_shot)

        clf.fit(z_support, y_support)

        return torch.from_numpy(clf.predict(z_query))
