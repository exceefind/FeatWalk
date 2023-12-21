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

# class Rec_Net(nn.Module):
#     def __init__(self,reduce_dim=128):
#         super(Rec_Net, self).__init__()
#         self.rec_layer = reconstruct_layer(in_channels=reduce_dim, out_channels=reduce_dim, ).cuda()
#         self.SFC = nn.Linear(reduce_dim, 5).cuda()
#         self.drop = nn.Dropout(0.5)
#         self.embeding_way='GE'
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self,sample_support,sample_support_cons,is_query=False):
#         rec_map = self.rec_layer(sample_support)
#         rec_map_cons = self.rec_layer(sample_support_cons)
#         # ==============================
#
#         if self.embeding_way in ['BDC']:
#             BDC_rec = self.dcov(rec_map)
#             BDC_rec_cons = self.dcov(rec_map_cons)
#
#         else:
#             BDC_rec = self.avg_pool(rec_map).view(sample_support.shape[0], -1)
#             BDC_rec_cons = self.avg_pool(rec_map_cons).view(sample_support.shape[0], -1)
#
#         spt_norm = torch.norm(BDC_rec, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec)
#         BDC_norm = BDC_rec.div(spt_norm + 1e-6)
#
#         spt_norm = torch.norm(BDC_rec_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec_cons)
#         BDC_norm_cons = BDC_rec_cons.div(spt_norm + 1e-6)
#         if is_query:
#
#             BDC_x = 0.5 * BDC_norm + 0.5 * (
#                         1 / 4 * torch.sum(BDC_norm_cons.view(BDC_norm.shape[0],-1,BDC_norm.shape[-1]), dim=1))
#
#         else:
#             BDC_x = (BDC_norm + BDC_norm_cons) / 2
#             BDC_x = self.drop(BDC_x)
#         out = self.SFC(BDC_x)
#         return out


class Net_rec(nn.Module):
    def __init__(self,params,num_classes = 5,):
        super(Net_rec, self).__init__()

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
        self.win_resize = False
        self.move_resize = False
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

    def comp_relation(self,feat_map,out_map =False):
    #     batch * feat_dim * feat_map
        batchSize, dim, h, w = feat_map.shape
        feat_map = feat_map.view(batchSize, dim, -1)
        # feat_map= self.drop_confusion(feat_map)
        feat_map_1 = feat_map.unsqueeze(-2).repeat(1,1,feat_map.shape[1],1)
        feat_map_2 = feat_map.unsqueeze(-2).permute(0,2,1,3).repeat(1,feat_map.shape[1],1,1)

        if self.params.lego :
            item_1 = torch.abs(feat_map_1 + feat_map_2)
            item_2 = torch.abs(feat_map_1 - feat_map_2)
            out = (item_1 - item_2) / 2
            out = out.reshape(batchSize, dim * dim)
            I = torch.ones(dim, dim).triu().reshape(dim * dim)
            index = I.nonzero(as_tuple=False)
            feat_map = out[:, index,:]
            feat_map_1 = feat_map.unsqueeze(-2).repeat(1, 1, feat_map.shape[1], 1)
            feat_map_2 = feat_map.unsqueeze(-2).permute(0, 2, 1, 3).repeat(1, feat_map.shape[1], 1, 1)
            item_1 = torch.sum(torch.abs(feat_map_1 + feat_map_2), dim=-1)
            item_2 = torch.sum(torch.abs(feat_map_1 - feat_map_2), dim=-1)
            out = (item_1 - item_2) * torch.exp(self.temperature)

        else:

            item_1 = torch.sum(torch.abs(feat_map_1 + feat_map_2), dim=-1)
            item_2 = torch.sum(torch.abs(feat_map_1 - feat_map_2), dim=-1)

            out = (item_1 - item_2)/2 * torch.exp(self.temperature)


        # out = torch.clamp(out, min=1e-8)

        # ==================================
        if self.params.normalize_bdc:
            I_M = torch.ones(batchSize, dim, dim, device=feat_map.device).type(feat_map.dtype)
            # out = out - 1. / dim * out.bmm(I_M) - 1. / dim * I_M.bmm(out) + 1. / (dim * dim) * I_M.bmm(out).bmm(I_M)
            out = out - 1. / dim * out.bmm(I_M) - 1. / dim * I_M.bmm(out)

        if out_map:
            out = Triumap(out.reshape(batchSize,dim,dim,self.feat_dim[-2],self.feat_dim[-1]), no_diag=self.params.no_diag)
            out = self.avg_pool(out)
        else:
            out = Triuvec(out,no_diag=self.params.no_diag)

        if self.params.normalize_feat:
            out = self.normalize(out)

        return out

    def forward_feature(self, x, confusion=False, out_map=False):
        feat_map = self.backbone(x, )
        if self.resnet_layer_dim[-1] != self.reduce_dim:
            # print('**************')
            feat_map = self.Conv(feat_map)
        # if self.params.LR:
        #     if self.params.embeding_way in ['BDC'] :
        #         out = self.dcov(feat_map)
        #     else:
        #         x = self.avg_pool(feat_map)
        #         out = x.view(x.shape[0], -1)
        # else:
        #     out = feat_map
        out = feat_map
        return out

    def normalize_feature(self, x):
        if self.params.norm == 'center':
            x = x - x.mean(2).unsqueeze(2)
            return x
        else:
            return x

    def forward_feat(self,x,x_c, confusion=False,):
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

    def forward_pretrain(self, x, confusion = False):
        x = self.forward_feature(x,confusion=confusion,out_map=False)

        # x = self.comp_relation(x)
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

        # print('k between {:d} and {:d}'.format(min(k_rocord),max(k_rocord),))
        # print('k between {:d} and {:d}'.format(min(k_rocord),max(k_rocord),))
        # print(Counter(k_rocord).most_common(4))
        return avg_loss / iter_num, float(total_correct) / total * 100

    def meta_val_loop(self,epoch,val_loader,classifier='emd'):
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i, data in enumerate(val_loader):
            tic = time.time()
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            split_size = 128
            if support_xs.squeeze(0).shape[0] >= split_size:
                feat_sup_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(support_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    fest_sup_item = self.forward_feature(
                        support_xs.squeeze(0)[j * split_size:min((j + 1) * split_size, support_xs.shape[1]), :, :, :],
                        out_map=self.out_map)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape) >= 1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_, dim=0)
            else:
                feat_sup = self.forward_feature(support_xs.squeeze(0), out_map=self.out_map)
            if query_xs.squeeze(0).shape[0] > split_size:
                feat_qry_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(query_xs.squeeze(0).shape[0] / split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
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
                # pred = pred.view(-1, self.params.n_symmetry_aug)
                query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
                # pred = torch.mode(pred,dim=-1)[0]
                query_ys = torch.mode(query_ys, dim=-1)[0]

            # print(np.mean(pred.cpu().numpy() == query_ys.numpy()))
            acc_epo = np.mean(pred.cpu().numpy() == query_ys.numpy())
            # if acc_epo<0.78:
            #     acc_task = []
            #     recall = []
            #     for i in range(self.params.n_way):
            #         acc_task.append(np.mean((pred.cpu().numpy()==i) & (query_ys.numpy()==i))*self.params.n_way)
            #         recall.append(np.mean((pred.cpu().numpy() == i) & (query_ys.numpy() != pred.cpu().numpy())) * self.params.n_way)
            #     print(acc_task,recall)
            #     time.sleep(100)
            acc.append(acc_epo)
            # print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
            #                                                                                      *mean_confidence_interval(
            #                                                                                          acc, multi=100), (
            #                                                                                              time.time() - tic) / 60),
            #       end='')
        return mean_confidence_interval(acc)

    def meta_test_loop(self,test_loader):
        acc = []
        classifier = 'emd' if not self.params.test_LR else 'LR'
        for i, (x, _) in enumerate(test_loader):
            self.params.n_aug_support_samples = self.transform_aug
            tic = time.time()
            # print(x.shape)
            x = x.contiguous().view(self.n_way, (self.n_shot + self.params.n_queries), *x.size()[2:])
            # print(x.shape)
            # if self.params.LR:
            #     support_xs = x[:, :self.n_shot].contiguous().view( self.n_way * self.n_shot , self.params.n_aug_support_samples, *x.size()[3:])[:,0].cuda()
            #     query_xs = x[:, self.n_shot:,0:self.params.n_symmetry_aug].contiguous().view(self.n_way * self.params.n_queries , self.params.n_symmetry_aug,*x.size()[3:])[:,0].cuda()
            # else:
            #     support_xs = x[:, :self.n_shot].contiguous().view(self.n_way * self.n_shot*self.params.n_aug_support_samples, *x.size()[3:]).cuda()
            #     query_xs = x[:, self.n_shot:,0:self.params.n_symmetry_aug].contiguous().view(self.n_way * self.params.n_queries*self.params.n_symmetry_aug, *x.size()[3:]).cuda()
            #     # print(query_xs.shape)

            support_xs = x[:, :self.n_shot].contiguous().view(
                self.n_way * self.n_shot * self.params.n_aug_support_samples, *x.size()[3:]).cuda()
            query_xs = x[:, self.n_shot:, 0:self.params.n_symmetry_aug].contiguous().view(
                self.n_way * self.params.n_queries * self.params.n_symmetry_aug, *x.size()[3:]).cuda()
            # print(query_xs.shape)

            if self.win_resize:
                level = 3
                assert self.params.n_aug_support_samples == 1
                support_xs = getWin_resize(support_xs, level=level)
                # print(support_xs.shape)
                support_xs = support_xs.view(-1, *x.size()[3:])
                query_xs = getWin_resize(query_xs, level=level).view(-1, *x.size()[3:])
                self.params.n_aug_support_samples = sum([i ** 2 for i in range(1, level + 1)])

            if self.move_resize:
                assert self.params.n_aug_support_samples == 1
                support_xs = getWin_resize_move(support_xs,).view(-1, *x.size()[3:])
                # print(support_xs.shape)
                query_xs = getWin_resize_move(query_xs,).view(-1, *x.size()[3:])
                self.params.n_aug_support_samples = support_xs.shape[0]//(self.n_way*self.n_shot)
                # print(self.params.n_aug_support_samples)


            support_y = torch.from_numpy(np.repeat(range(self.params.n_way),self.n_shot*self.params.n_aug_support_samples)).unsqueeze(0)
            # query_ys = torch.from_numpy(np.repeat(range(self.params.n_way),self.params.n_queries*self.params.n_aug_support_samples)).unsqueeze(0)

            # print(query_xs.shape)
            split_size = 512
            if support_xs.shape[0] >= split_size:
                feat_sup_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(support_xs.shape[0]/split_size)):
                    # print(support_xs.squeeze(0)[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    fest_sup_item =self.forward_feature(support_xs[j*split_size:min((j+1)*split_size,support_xs.shape[0]),],out_map=self.out_map)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape)>=1 else fest_sup_item.unsqueeze(0))
                feat_sup = torch.cat(feat_sup_,dim=0)
            else:
                feat_sup = self.forward_feature(support_xs,out_map=self.out_map)
            if query_xs.shape[0] >= split_size:
                feat_qry_ = []
                # print(support_xs.shape)
                for j in range(math.ceil(query_xs.shape[0]/split_size)):
                    # print(support_xs[i*128:min((i+1)*128,support_xs.shape[1]),:,:,:].shape)
                    feat_qry_item = self.forward_feature(
                        query_xs[j * split_size:min((j + 1) * split_size, query_xs.shape[0]), ],out_map=self.out_map)
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))

                feat_qry = torch.cat(feat_qry_,dim=0)
            else:
                feat_qry = self.forward_feature(query_xs,out_map=self.out_map)


            if self.params.softmax_aug:
                # print("--11-")
                with torch.enable_grad():
                    pred = self.softmax_aug(feat_sup, support_y, feat_qry,)
                    _,pred = torch.max(pred,dim=-1)
            elif self.params.embeding_way in ['protonet']:
                # print(support_y)
                pred = self.predict_wo_fc(feat_sup, support_y, feat_qry,)
                _, pred = torch.max(pred, dim=-1)
            elif self.params.LR or self.params.more_classifier is not None:
                pred = self.predict_wo_fc(feat_sup, support_y, feat_qry,)
            elif self.params.same_computation :
                with torch.enable_grad():
                    pred = self.softmax_sc(feat_sup, support_y, feat_qry,)
                    if not self.params.LR_rec:
                        _,pred = torch.max(pred,dim=-1)
            else:
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_y, feat_qry,)
                    if not self.params.LR_rec:
                        _,pred = torch.max(pred,dim=-1)
            # if self.params.n_symmetry_aug > 1:
            #     # pred = pred.view(-1, self.params.n_symmetry_aug)
            #     query_ys = query_ys.view(-1, self.params.n_symmetry_aug)
            #     # pred = torch.mode(pred,dim=-1)[0]
            #     query_ys = torch.mode(query_ys, dim=-1)[0]


            query_ys = np.repeat(range(self.n_way), self.params.n_queries)
            # print(pred.shape)
            # print(query_ys.shape)
            # print(np.mean(pred.cpu().numpy() == query_ys.numpy()))
            # print(query_ys)
            pred = pred.view(-1)

            acc_epo = np.mean(pred.cpu().numpy() == query_ys)
            # if acc_epo<0.78:
            #     acc_task = []
            #     recall = []
            #     for i in range(self.params.n_way):
            #         acc_task.append(np.mean((pred.cpu().numpy()==i) & (query_ys.numpy()==i))*self.params.n_way)
            #         recall.append(np.mean((pred.cpu().numpy() == i) & (query_ys.numpy() != pred.cpu().numpy())) * self.params.n_way)
            #     print(acc_task,recall)
            #     time.sleep(100)
            acc.append(acc_epo)
            print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
                                                                                                 *mean_confidence_interval(
                                                                                                     acc, multi=100), (
                                                                                                             time.time() - tic) / 60),
                  end='')

        # np.save("train_loss2.npy",np.array(train_loss_list))
        # np.save("test_loss2.npy",np.array(test_loss_list))
        # np.save("proto_changes2.npy",np.array(prototypes_changes_list))
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

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score

    # new selective local fusion :
    def softmax(self,support_z,support_ys,query_z,):
        # print("--110-")
        loss_ce_fn = nn.CrossEntropyLoss()
        lr_scheduler = None
        batch_size = self.params.sfc_bs
        walk_times = 24
        alpha = self.params.alpha
        tempe = self.params.sim_temperature

        # feat_a = []
        # feat_b = []
        # label_c = []
        # train_loss = []
        # test_loss = []
        # prototypes_change = []

        support_ys = support_ys.cuda()

        if self.params.embeding_way in ['BDC']:
            SFC = nn.Linear(self.dim, self.params.n_way).cuda()

            # tempe = torch.tensor([64.],requires_grad=True,device=support_z.device)
            # tempe = torch.tensor([-.1],requires_grad=True,device=support_z.device)
            # fusion =fusion_module(dim=self.dim).cuda()
            if self.params.optim in ['Adam']:
                optimizer = torch.optim.Adam([{'params': SFC.parameters()}],lr=self.params.lr, weight_decay=self.params.wd_test)
                iter_num = 100

                # optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                #                              weight_decay=0.001,eps=1e-4)
                optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                                             weight_decay=self.params.wd_test,eps=1e-4)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_num * math.ceil(self.n_way*self.n_shot/batch_size),eta_min=1e-3)

                # optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                #                               weight_decay=self.params.wd_test, eps=5e-5)
                # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_num * math.ceil(
                #     self.n_way * self.n_shot / batch_size), eta_min=1e-3)
            else:
                # best
                # optimizer = torch.optim.SGD([{'params':fusion.parameters(),'lr':self.params.lr},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9,  weight_decay=self.params.wd_test)
                optimizer = torch.optim.SGD([{'params': SFC.parameters()},{'params':tempe,'lr':0.0}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=self.params.wd_test)
                # 1shot : 69.20+-    5shot 85.73
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.1)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
                iter_num = 180
        else:
            tempe =16

            # tempe = 4

            if self.params.embeding_way in ['baseline++']:
                SFC = nn.Linear(self.reduce_dim, self.params.n_way, bias=False).cuda()
                WeightNorm.apply(SFC, 'weight', dim=0)
            else:
                SFC = nn.Linear(self.reduce_dim, self.params.n_way).cuda()

            if self.params.optim in ['Adam']:
                # lr = 5e-3
                optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.005,
                                              weight_decay=self.params.wd_test, eps=5e-3)
                # optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                #                               weight_decay=self.params.wd_test, eps=5e-3)
                iter_num = 100
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_num * math.ceil(
                    self.n_way * self.n_shot / batch_size), eta_min=5e-3)


            else:
                optimizer = torch.optim.SGD([{'params': SFC.parameters()}],
                                            lr=self.params.lr, momentum=0.9, nesterov=True,
                                            weight_decay=self.params.wd_test)

                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
                iter_num = 180

                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
                # iter_num = 150

                # 1shot : 69.20+-    5shot 85.73
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
                # iter_num = 180
                # 62.05 77.54
                # 62.05 77.54
                # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), 'lr': 2}, {'params': SFC.parameters()}],
                #                             lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # optimizer = torch.optim.SGD([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # 1-shot pure : 62.57
                # optimizer = torch.optim.SGD([{'params':rec_layer.parameters(),'lr':5e-1},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=5e-2)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600,800], gamma=0.1)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 240], gamma=0.2)
                #
                # iter_num = 360
            # rec_layer = reconstruct_layer().cuda()
            # SFC = nn.Linear(self.dim, self.params.n_way).cuda()

            # self.drop = nn.Dropout(0.6)
            # Good Embedding
            # 62.4
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr':5e-2},{'params': SFC.parameters()}],lr=4e-3, weight_decay=1e-4,eps=1e-5)
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr': 1e-3},{'params': SFC.parameters()}],lr=1e-3, weight_decay=1e-4)
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=5e-3, weight_decay=5e-4)

        SFC.train()

        if self.params.embeding_way in ['BDC']:
            support_z = self.dcov(support_z)
            query_z = self.dcov(query_z)

        else:
            support_z = self.avg_pool(support_z).view(support_z.shape[0], -1)
            query_z = self.avg_pool(query_z)

        support_ys = support_ys.view(self.n_way * self.n_shot, self.params.n_aug_support_samples, -1)
        global_ys = support_ys[:, 0, :]

        # spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        # support_z = support_z.div(spt_norm + 1e-6)
        # spt_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        # query_z = query_z.div(spt_norm + 1e-6)

        support_z = support_z.reshape(self.n_way,self.n_shot,self.params.n_aug_support_samples,-1)
        query_z = query_z.reshape(self.n_way,self.params.n_queries,self.params.n_aug_support_samples,-1)

        # print(support_z.shape)

        feat_q = query_z[:,:,0]
        feat_ql = query_z[:,:,1:]
        feat_g = support_z[:,:,0]
        feat_sl = support_z[:,:,1:]
        # w_local: n * k * n * m
        num_sample = self.n_way*self.n_shot
        global_ys = global_ys.view(self.n_way,self.n_shot,-1)

        feat_g = feat_g.detach()
        feat_sl = feat_sl.detach()

        # feat_a.append(feat_g.view(-1,self.dim).detach().cpu().numpy())
        # feat_a.append(feat_q.view(-1,self.dim).detach().cpu().numpy())
        # feat_a = np.concatenate(feat_a)
        # label_c.append(global_ys.view(-1).detach().cpu().numpy())
        # label_c.append(np.repeat(range(self.n_way), self.params.n_queries))
        # label_c = np.concatenate(label_c)
        # np.save("feat_a",feat_a)
        # np.save("label",label_c)

        # feat_sl: n * k * n *  dim

        # support_x = 0.5*(feat_g.unsqueeze(-2) + feat_s)
        I = torch.eye(self.n_way,self.n_way,device=feat_g.device).unsqueeze(0).unsqueeze(1)
        proto_moving = torch.mean(feat_g, dim=1)
        # proto_moving = torch.rand_like(torch.mean(feat_g, dim=1))
        # proto = torch.zeros((self.n_way, self.dim), device=support_z.device)



        for i in range(iter_num):
            weight = compute_weight_local(proto_moving.unsqueeze(1), feat_sl, feat_sl, self.params.measure)
            idx_walk = torch.randperm(self.params.n_aug_support_samples-1,)[:walk_times]
            w_local = F.softmax(weight[:,:,:,idx_walk] * tempe, dim=-1)
            # print(w_local.shape)
            feat_s = torch.sum((feat_sl[:,:,idx_walk,:].unsqueeze(-3)) * (w_local.detach().unsqueeze(-1)), dim=-2)
            # w_local = F.softmax((w_local * 1.5) / tempe, dim=-1)
            # feat_s = torch.sum(feat_sl.unsqueeze(-3) * w_local.detach().unsqueeze(-1), dim=-2)
            support_x = alpha * feat_g.unsqueeze(-2) + (1- alpha) * feat_s
            # spt_norm = torch.norm(support_x, p=2, dim=-1).unsqueeze(-1).expand_as(support_x)
            # support_x = support_x.div(spt_norm + 1e-6)
            proto_update = torch.sum(torch.matmul(torch.mean(support_x,dim=1).transpose(1,2),torch.eye(self.n_way,device=proto_moving.device).unsqueeze(0)),dim=-1)

            # prototypes_change.append(torch.mean((proto_moving-proto_update)**2).item())

            proto_moving = 0.9 * proto_moving + 0.1 * proto_update
            # proto_moving = proto_update
            # print(support_x.shape)
            # if i == iter_num-1:
            #     feat_b.append(support_x.view(-1,self.n_way,self.dim)[range(self.n_way*self.n_shot),global_ys.view(-1)].detach().cpu().numpy())

            # support_x = feat_g.unsqueeze(-2) + torch.randint(1,(self.n_way,self.n_shot,1),device=feat_g.device)*feat_s

            # print(sample_idxs)
            # sample_idxs = torch.arange(0,num_sample)
        # support_x = feat_g.unsqueeze(-2)
            spt_norm = torch.norm(support_x, p=2, dim=-1).unsqueeze(-1).expand_as(support_x)
            support_x = support_x.div(spt_norm + 1e-6)


        for i in range(iter_num):
            SFC.train()
            sample_idxs = torch.randperm(num_sample)
            # train_loss.append(0)
            for j in range(math.ceil(num_sample/batch_size)):
                idxs = sample_idxs[j*batch_size:min((j+1)*batch_size,num_sample)]
                x =  support_x[idxs//self.n_shot,idxs%self.n_shot]
                # print(idxs//self.n_way,idxs%self.n_way)
                y = global_ys[idxs//self.n_shot,idxs%self.n_shot]
                # print(y.view(-1))
                # print(x.shape)
                # print(global_ys.shape)
                x = self.drop(x)
                # out = torch.mean(SFC(x),dim=-2).view(-1,self.n_way)
                out = torch.sum(SFC(x)*I,dim=-1).view(-1,self.n_way)
                # print(out)
                # print(out.shape)
                # print(global_ys.shape)
                loss_ce = loss_ce_fn(out,y.long().view(-1))
                # train_loss[-1] += loss_ce.item()
                loss = loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            # print('loss_ce: {:.2f} \t loss_mse: {:.2f}'.format(loss_ce,loss_rec))
            # train_loss[-1] /= math.ceil(num_sample/batch_size)

            SFC.eval()

            # print(tempe.data)
            # print(proto_moving.shape)
            # print(feat_ql.shape)
            # print(feat_sl.shape)

        w_local = compute_weight_local(proto_moving.unsqueeze(1), feat_ql, feat_sl,self.params.measure)
        w_local = F.softmax(w_local *  tempe, dim=-1)

        # feat_sl: n * k * n *  dim
        feat_lq = torch.sum(feat_ql.unsqueeze(-3) * w_local.unsqueeze(-1), dim=-2)
        query_x =  alpha * feat_q.unsqueeze(-2) + (1- alpha) * feat_lq
        # query_x = feat_q.unsqueeze(-2)

        # feat_b.append(query_x.view(-1, self.n_way, self.dim)[
        #                   range(self.n_way * self.params.n_queries), np.repeat(range(self.n_way), self.params.n_queries)].detach().cpu().numpy())
        # feat_b = np.concatenate(feat_b)
        # np.save("feat_b",feat_b)

        spt_norm = torch.norm(query_x, p=2, dim=-1).unsqueeze(-1).expand_as(query_x)
        query_x = query_x.div(spt_norm + 1e-6)
        y_query = torch.tensor(np.repeat(range(self.params.n_way), self.params.n_queries)).cuda()
        with torch.no_grad():

            # out = torch.mean(SFC(query_x),dim=-2).view(-1,self.n_way)
            out = torch.sum(SFC(query_x)*I,dim=-1).view(-1,self.n_way)
            # test_loss.append(loss_ce_fn(out,y_query.long().view(-1)).item())

        # train_loss_list.append(train_loss)
        # test_loss_list.append(test_loss)
        # prototypes_changes_list.append(prototypes_change)
        # print(train_loss)
        # print(test_loss)
        # print(prototypes_change)
        return out

    def softmax_sc(self,support_z,support_ys,query_z,):
        # print("--110-")
        loss_ce_fn = nn.CrossEntropyLoss()
        batch_size = 4
        alpha = self.params.alpha
        tempe = self.params.sim_temperature

        support_ys = support_ys.cuda()

        if self.params.embeding_way in ['BDC']:
            if self.params.same_computation == 'cat':
                SFC = nn.Linear(self.dim * self.params.n_aug_support_samples, self.params.n_way).cuda()
            else:
                SFC = nn.Linear(self.dim, self.params.n_way).cuda()
            if self.params.optim in ['Adam']:
                optimizer = torch.optim.Adam([{'params': SFC.parameters()}],lr=self.params.lr, weight_decay=self.params.wd_test)
                iter_num = 100

                # optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                #                              weight_decay=0.001,eps=1e-4)
                optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                                             weight_decay=self.params.wd_test,eps=1e-4)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_num * math.ceil(self.n_way*self.n_shot/batch_size),eta_min=1e-3)

            else:
                # best
                # optimizer = torch.optim.SGD([{'params':fusion.parameters(),'lr':self.params.lr},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9,  weight_decay=self.params.wd_test)
                optimizer = torch.optim.SGD([{'params': SFC.parameters()},{'params':tempe,'lr':0.0}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=self.params.wd_test)
                # 1shot : 69.20+-    5shot 85.73
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.1)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
                iter_num = 180
        else:
            tempe =16

            # tempe = 4

            if self.params.embeding_way in ['baseline++']:
                SFC = nn.Linear(self.reduce_dim, self.params.n_way, bias=False).cuda()
                WeightNorm.apply(SFC, 'weight', dim=0)
            else:
                if self.params.same_computation == 'cat':
                    SFC = nn.Linear(self.reduce_dim * self.params.n_aug_support_samples, self.params.n_way).cuda()
                else:
                    SFC = nn.Linear(self.reduce_dim, self.params.n_way).cuda()
                # SFC = nn.Linear(self.reduce_dim, self.params.n_way).cuda()

            if self.params.optim in ['Adam']:
                # lr = 5e-3
                optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.005,
                                              weight_decay=self.params.wd_test, eps=5e-3)
                # optimizer = torch.optim.AdamW([{'params': SFC.parameters()}], lr=0.001,
                #                               weight_decay=self.params.wd_test, eps=5e-3)
                iter_num = 100
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_num * math.ceil(
                    self.n_way * self.n_shot / batch_size), eta_min=5e-3)


            else:
                optimizer = torch.optim.SGD([{'params': SFC.parameters()}],
                                            lr=self.params.lr, momentum=0.9, nesterov=True,
                                            weight_decay=self.params.wd_test)

                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
                iter_num = 180

                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
                # iter_num = 150

                # 1shot : 69.20+-    5shot 85.73
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
                # iter_num = 180
                # 62.05 77.54
                # 62.05 77.54
                # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), 'lr': 2}, {'params': SFC.parameters()}],
                #                             lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # optimizer = torch.optim.SGD([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
                # # 1-shot pure : 62.57
                # optimizer = torch.optim.SGD([{'params':rec_layer.parameters(),'lr':5e-1},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=5e-2)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600,800], gamma=0.1)
                # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 240], gamma=0.2)
                #
                # iter_num = 360
            # rec_layer = reconstruct_layer().cuda()
            # SFC = nn.Linear(self.dim, self.params.n_way).cuda()

            # self.drop = nn.Dropout(0.6)
            # Good Embedding
            # 62.4
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr':5e-2},{'params': SFC.parameters()}],lr=4e-3, weight_decay=1e-4,eps=1e-5)
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr': 1e-3},{'params': SFC.parameters()}],lr=1e-3, weight_decay=1e-4)
            # optimizer = torch.optim.Adam([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=5e-3, weight_decay=5e-4)

        SFC.train()

        if self.params.embeding_way in ['BDC']:
            support_z = self.dcov(support_z)
            query_z = self.dcov(query_z)

        else:
            support_z = self.avg_pool(support_z).view(support_z.shape[0], -1)
            query_z = self.avg_pool(query_z)

        support_ys = support_ys.view(self.n_way * self.n_shot, self.params.n_aug_support_samples, -1)
        global_ys = support_ys[:, 0, :]

        # spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        # support_z = support_z.div(spt_norm + 1e-6)
        # spt_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        # query_z = query_z.div(spt_norm + 1e-6)


        support_z = support_z.reshape(self.n_way,self.n_shot,self.params.n_aug_support_samples,-1)
        query_z = query_z.reshape(self.n_way,self.params.n_queries,self.params.n_aug_support_samples,-1)

        # print(support_z.shape)

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

        # support_x = 0.5*(feat_g.unsqueeze(-2) + feat_s)
        I = torch.eye(self.n_way,self.n_way,device=feat_g.device).unsqueeze(0).unsqueeze(1)
        proto_moving = torch.mean(feat_g, dim=1)
        # proto = torch.zeros((self.n_way, self.dim), device=support_z.device)

        spt_norm = torch.norm(support_z, p=2, dim=-1).unsqueeze(-1).expand_as(support_z)
        support_x = support_z.div(spt_norm + 1e-6)



        for i in range(iter_num):
            sample_idxs = torch.randperm(num_sample)
            support_x = support_x.detach()
            for j in range(math.ceil(num_sample/batch_size)):
                idxs = sample_idxs[j*batch_size:min((j+1)*batch_size,num_sample)]
                x =  support_x[idxs//self.n_shot,idxs%self.n_shot]
                # print(idxs//self.n_way,idxs%self.n_way)
                y = global_ys[idxs//self.n_shot,idxs%self.n_shot]
                if self.params.same_computation == 'cat':
                    x = x.reshape(x.shape[0],-1)
                    x = self.drop(x)
                    out = SFC(x).view(-1,self.n_way)
                else:
                    if self.params.same_computation == 'mean':
                        x = self.drop(x)
                        out = torch.mean(SFC(x), dim=-2).view(-1, self.n_way)
                    elif self.params.same_computation == 'max_late':
                        x = torch.max(x,dim=-2)[0]
                        x = self.drop(x)
                        out = SFC(x).view(-1, self.n_way)
                    else:
                        x = self.drop(x)
                        out = torch.max(SFC(x), dim=-2)[0].view(-1, self.n_way)

                loss_ce = loss_ce_fn(out,y.long().view(-1))
                loss = loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            # print('loss_ce: {:.2f} \t loss_mse: {:.2f}'.format(loss_ce,loss_rec))


        SFC.eval()



        spt_norm = torch.norm(query_z, p=2, dim=-1).unsqueeze(-1).expand_as(query_z)
        query_x = query_z.div(spt_norm + 1e-6)


        with torch.no_grad():
            if self.params.same_computation == 'cat':
                # print(query_x.shape)
                query_x = query_x.reshape(query_x.shape[0]*query_x.shape[1], -1)
                out = SFC(query_x).view(-1, self.n_way)
            else:
                if self.params.same_computation == 'mean':
                    out = torch.mean(SFC(query_x), dim=-2).view(-1, self.n_way)
                elif self.params.same_computation == 'max_late':
                    query_x = torch.max(query_x, dim=-2)[0]
                    out = SFC(query_x).view(-1, self.n_way)
                else:
                    out = torch.max(SFC(query_x), dim=-2)[0].view(-1, self.n_way)
            # out = torch.sum(SFC(query_x)*I,dim=-1).view(-1,self.n_way)

        return out

    # def softmax(self,support_z,support_ys,query_z,):
    #     # proto : K * D
    #     prototype = torch.zeros((self.params.n_way, self.dim)).cuda()
    #     # prototype = torch.zeros((support_z.shape[0]//self.params.n_aug_support_samples, self.dim)).cuda()
    #     loss_ce_fn = nn.CrossEntropyLoss()
    #     lr_scheduler = None
    #     support_ys = support_ys.cuda()
    #     drop2 = nn.Dropout(0.3).cuda()
    #
    #     if self.params.embeding_way in ['BDC']:
    #         rec_layer = reconstruct_layer().cuda()
    #         SFC = nn.Linear(self.dim, self.params.n_way).cuda()
    #         fusion =fusion_module(dim=self.dim).cuda()
    #         if self.params.optim in ['Adam']:
    #             optimizer = torch.optim.Adam([{'params':fusion.parameters()},{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=self.params.lr, weight_decay=self.params.wd_test)
    #             iter_num = 100
    #         else:
    #             # best
    #             # optimizer = torch.optim.SGD([{'params':fusion.parameters(),'lr':self.params.lr},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9,  weight_decay=self.params.wd_test)
    #             optimizer = torch.optim.SGD([{'params':fusion.parameters(),'lr':self.params.lr},{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=self.params.wd_test)
    #             # 1shot : 69.20+-    5shot 85.73
    #             lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,150], gamma=0.1)
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
    #             iter_num = 180
    #     else:
    #
    #         rec_layer = reconstruct_layer(in_channels=self.reduce_dim, out_channels=self.reduce_dim,).cuda()
    #         if self.params.embeding_way in ['baseline++']:
    #             SFC = nn.Linear(self.reduce_dim, self.params.n_way, bias=False).cuda()
    #             WeightNorm.apply(SFC, 'weight', dim=0)
    #         else:
    #             SFC = nn.Linear(self.reduce_dim, self.params.n_way).cuda()
    #         fusion =fusion_module(dim=self.reduce_dim).cuda()
    #         if self.params.optim in ['Adam']:
    #             # lr = 5e-3
    #             optimizer = torch.optim.Adam([{'params':fusion.parameters()},{'params': rec_layer.parameters()}, {'params': SFC.parameters()}],
    #                                          lr=self.params.lr, weight_decay=self.params.wd_test)
    #             # optimizer = torch.optim.Adam([{'params': SFC.parameters()}],lr=1e-3, weight_decay=5e-4)
    #             #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,], gamma=0.5)
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, ], gamma=0.2)
    #             # iter_num = 150
    #
    #             optimizer = torch.optim.Adam([{'params': rec_layer.parameters()}, {'params': SFC.parameters()}],
    #                                          lr=self.params.lr, weight_decay=self.params.wd_test)
    #             iter_num = 100
    #
    #         else:
    #             optimizer = torch.optim.SGD([{'params':fusion.parameters()},{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],
    #                                         lr=self.params.lr, momentum=0.9, nesterov=True,
    #                                         weight_decay=self.params.wd_test)
    #             # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), }, {'params': SFC.parameters()}],
    #             #                             lr=self.params.lr, momentum=0.9,
    #             #                             weight_decay=self.params.wd_test)
    #             # optimizer = torch.optim.LBFGS([{'params': rec_layer.parameters(), }, {'params': SFC.parameters()}],lr=self.params.lr,
    #             #                               )
    #             lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,160,240], gamma=0.1)
    #
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 240, 360],
    #             #                                                     gamma=0.1)
    #             # iter_num = 450
    #             iter_num = 300
    #             # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=iter_num,eta_min=1e-4)
    #
    #             lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
    #             iter_num = 180
    #
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)
    #             # iter_num = 150
    #
    #             # 1shot : 69.20+-    5shot 85.73
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 150], gamma=0.1)
    #             # iter_num = 180
    #             # 62.05 77.54
    #             # 62.05 77.54
    #             # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(), 'lr': 2}, {'params': SFC.parameters()}],
    #             #                             lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
    #             # # optimizer = torch.optim.SGD([{'params': rec_layer.parameters(),}, {'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
    #             # # optimizer = torch.optim.SGD([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.05)
    #             # # 1-shot pure : 62.57
    #             # optimizer = torch.optim.SGD([{'params':rec_layer.parameters(),'lr':5e-1},{'params': SFC.parameters()}],lr=self.params.lr, momentum=0.9, nesterov=True, weight_decay=5e-2)
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600,800], gamma=0.1)
    #             # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 240], gamma=0.2)
    #             #
    #             # iter_num = 360
    #         # rec_layer = reconstruct_layer().cuda()
    #         # SFC = nn.Linear(self.dim, self.params.n_way).cuda()
    #
    #         # self.drop = nn.Dropout(0.6)
    #         # Good Embedding
    #         # 62.4
    #         # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr':5e-2},{'params': SFC.parameters()}],lr=4e-3, weight_decay=1e-4,eps=1e-5)
    #         # optimizer = torch.optim.Adam([{'params':rec_layer.parameters(),'lr': 1e-3},{'params': SFC.parameters()}],lr=1e-3, weight_decay=1e-4)
    #         # optimizer = torch.optim.Adam([{'params':rec_layer.parameters()},{'params': SFC.parameters()}],lr=5e-3, weight_decay=5e-4)
    #
    #     # loss_rec = 0
    #     rec_layer.train()
    #     SFC.train()
    #     batch_fsl =min(16,support_z.shape[0])
    #     no_stym_aug = False
    #     sam_local = True
    #     local_num = 5
    #
    #     for i in range(iter_num):
    #         # sample_idxs = np.random.choice(range(support_z.shape[0]),min(64, support_z.shape[0]))
    #         # sample_idxs = random_sample(self.params.n_aug_support_samples,support_z.shape[0],self.params.n_way*self.params.n_aug_support_samples)
    #         # sample_idxs = random_sample(self.params.n_aug_support_samples,support_z.shape[0],min(25, self.params.n_way*self.params.n_shot))
    #
    #         sample_idxs = range(0,support_z.shape[0],self.params.n_aug_support_samples)
    #         if sam_local:
    #             sample_idxs_cons = []
    #             for s in range(local_num):
    #                 sample_cons_part = torch.tensor(random_sample(self.params.n_aug_support_samples, support_z.shape[0],
    #                                                  support_z.shape[0] // self.params.n_aug_support_samples))
    #                 sample_idxs_cons.append(sample_cons_part.unsqueeze(0))
    #             sample_idxs_cons = torch.cat(sample_idxs_cons).view(-1)
    #         else:
    #             sample_idxs_cons = random_sample(self.params.n_aug_support_samples,support_z.shape[0],support_z.shape[0]//self.params.n_aug_support_samples)
    #         #
    #         # if self.params.embeding_way in ['baseline++']:
    #         #     sample_idxs = np.random.choice(range(support_z.shape[0] // self.params.n_aug_support_samples),
    #         #                                    batch_fsl) * self.params.n_aug_support_samples
    #         #     sample_idxs_cons = sample_idxs + np.random.choice(range(self.params.n_aug_support_samples),batch_fsl)
    #
    #         sample_support = support_z[sample_idxs, :, :, :]
    #         sample_support_cons = support_z[sample_idxs_cons, :, :, :]
    #         sample_label = support_ys[:, sample_idxs]
    #
    #         # rec_map = rec_layer(sample_support)
    #         # rec_map_cons = rec_layer(sample_support_cons)
    #         # ==============================
    #
    #         if self.params.embeding_way in ['BDC']:
    #             BDC_ori = self.dcov(sample_support)
    #             BDC_ori_cons = self.dcov(sample_support_cons)
    #             # BDC_rec = self.dcov(rec_map)
    #             # BDC_rec_cons = self.dcov(rec_map_cons)
    #
    #         else:
    #             BDC_ori = self.avg_pool(sample_support).view(sample_support.shape[0],-1)
    #             BDC_ori_cons = self.avg_pool(sample_support_cons).view(sample_support_cons.shape[0],-1)
    #             # BDC_rec = self.avg_pool(rec_map).view(sample_support.shape[0],-1)
    #             # BDC_rec_cons = self.avg_pool(rec_map_cons).view(sample_support.shape[0],-1)
    #
    #         # BDC_ori = self.comp_relation(sample_support)
    #         spt_norm = torch.norm(BDC_ori, p=2, dim=1).unsqueeze(1).expand_as(BDC_ori)
    #         BDC_ori = BDC_ori.div(spt_norm + 1e-6 )
    #         #
    #         spt_norm_cons = torch.norm(BDC_ori_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_ori_cons)
    #         BDC_ori_cons = BDC_ori_cons.div(spt_norm_cons + 1e-6)
    #         #
    #         # # if not self.params.LR_rec:
    #         # spt_norm = torch.norm(BDC_rec, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec)
    #         # BDC_rec = BDC_rec.div(spt_norm + 1e-6)
    #         #
    #         # spt_norm = torch.norm(BDC_rec_cons, p=2, dim=1).unsqueeze(1).expand_as(BDC_rec_cons)
    #         # BDC_rec_cons = BDC_rec_cons.div(spt_norm + 1e-6)
    #
    #         if sam_local :
    #             # print(BDC_ori_cons.shape)
    #             BDC_ori_cons = torch.mean(BDC_ori_cons.view(local_num,-1,*BDC_ori_cons.size()[1:]),dim=0)
    #
    #         if self.params.n_aug_support_samples==1:
    #             BDC_x = BDC_ori
    #         else:
    #             BDC_x = (BDC_ori_cons + BDC_ori) / 2
    #         if np.random.rand() <= self.params.drop_few:
    #             # pass
    #             BDC_x = BDC_ori_cons
    #
    #         # BDC_x_norm = torch.norm(BDC_x, p=2, dim=1).unsqueeze(1).expand_as(BDC_x)
    #         # BDC_x = BDC_x.div(BDC_x_norm + 1e-6)
    #
    #         if self.params.ablation % 2 == 1:
    #             BDC_x = BDC_ori_cons
    #         if self.params.ablation % 2 == 0 and self.params.ablation >1:
    #             BDC_x = BDC_ori
    #
    #         BDC_x = self.drop(BDC_x)
    #         out = SFC(BDC_x)
    #         # out = SFC(BDC_rec)
    #         # if no_stym_aug:
    #         #     out = out.reshape(BDC_rec.shape[0], self.params.n_way, -1)
    #         #     # out = torch.max(out,dim=1)[0]
    #         #     # out = out - torch.mean(out,dim=1).unsqueeze(1) - torch.mean(out,dim=2).unsqueeze(2) + torch.mean(torch.mean(out,dim=-1),dim=-1).unsqueeze(-1).unsqueeze(-1)
    #         #     out =  out[:, range(self.params.n_way), range(self.params.n_way)]
    #
    #         loss_ce = loss_ce_fn(out,sample_label.squeeze(0))
    #
    #
    #         loss = loss_ce
    #         # loss = loss_rec
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # print(fusion.weight.data[0])
    #         if lr_scheduler is not None:
    #             lr_scheduler.step()
    #         # print('loss_ce: {:.2f} \t loss_mse: {:.2f}'.format(loss_ce,loss_rec))
    #
    #     rec_layer.eval()
    #     SFC.eval()
    #     fusion.eval()
    #     if self.params.LR_rec:
    #         support_z_rec = rec_layer(support_z)
    #
    #         if self.params.embeding_way in ['BDC']:
    #             support_z_rec = self.dcov(support_z_rec)
    #         else:
    #             support_z_rec = self.avg_pool(support_z_rec)
    #         spt_norm = torch.norm(support_z_rec, p=2, dim=1).unsqueeze(1).expand_as(support_z_rec)
    #         support_z_rec = support_z_rec.div(spt_norm + 1e-6)
    #         support_z_rec = support_z_rec.reshape(support_z.shape[0]//self.params.n_aug_support_samples,self.params.n_aug_support_samples,-1)
    #         # support_ys = support_ys.view((support_z.shape[0]//self.params.n_aug_support_samples,-1))[:,0]
    #         support_ys = support_ys.reshape((support_z.shape[0]//self.params.n_aug_support_samples,-1))[:,1:]
    #
    #         # support_z_rec = 0.5 * support_z_rec[:,0,:] + 0.5* torch.mean(support_z_rec[:,1:,:],dim=1)
    #         support_z_rec = 0.5 * support_z_rec[:,0,:].unsqueeze(1) + 0.5* support_z_rec[:,1:,:]
    #         # support_z_rec = support_z_rec[:,0,:]
    #         clf = self.LR_rec(
    #             support_z_rec.reshape(support_z_rec.shape[0] * (self.params.n_aug_support_samples - 1), -1),
    #             support_ys)
    #         # clf = self.LR_rec(
    #         #     support_z_rec,
    #         #     support_ys)
    #
    #     with torch.no_grad():
    #         # print(query_z.shape)
    #         # query_rec = rec_layer(query_z)
    #         # query_rec = query_z
    #         if self.params.embeding_way in ['BDC']:
    #             query_ori = self.dcov(query_z)
    #             # query_rec = self.dcov(query_rec)
    #         else:
    #             query_ori = self.avg_pool(query_z)
    #             # query_rec = self.avg_pool(query_rec)
    #             # query_rec = self.comp_relation(query_rec)
    #
    #         # query_rec = self.dcov(query_z)
    #         # query_rec = self.comp_relation(query_rec)
    #         # spt_norm = torch.norm(query_rec, p=2, dim=1).unsqueeze(1).expand_as(query_rec)
    #         # query_rec = query_rec.div(spt_norm + 1e-6)
    #
    #         spt_norm = torch.norm(query_ori, p=2, dim=1).unsqueeze(1).expand_as(query_ori)
    #         query_ori = query_ori.div(spt_norm + 1e-6)
    #         # print(query_rec.shape)
    #         # print(query_rec.shape)
    #         # query_rec = query_rec.view(query_rec.shape[0]//self.params.n_symmetry_aug, self.params.n_symmetry_aug, -1)
    #         # query_rec = query_rec.view(query_rec.shape[0]//self.params.n_symmetry_aug, self.params.n_symmetry_aug, -1)
    #         query_ori = query_ori.view(query_ori.shape[0]//self.params.n_symmetry_aug, self.params.n_symmetry_aug, -1)
    #
    #         # query_rec = 1 / (self.params.n_symmetry_aug) * torch.sum(query_rec, dim=1)
    #         if self.params.n_symmetry_aug>1:
    #             # query_rec = 1 / (self.params.n_symmetry_aug) * torch.sum(query_rec, dim=1)
    #             # query_rec = 0.5 * query_rec[:,0,:] + 0.5 * (1 / (self.params.n_symmetry_aug-1) * torch.sum(query_rec[:,1:,:], dim=1))
    #             # query_rec = 0.5 * query_ori[:,0,:] + 0.5 * (1 / (self.params.n_symmetry_aug-1) * torch.sum(query_rec[:,1:,:], dim=1))
    #             # query_rec = 0.5 * query_ori[:,0,:] + 0.5 * (1 / (self.params.n_symmetry_aug-1) * torch.sum(query_ori[:,1:self.params.n_symmetry_aug,:], dim=1))
    #             # query_cons = (1 / (self.params.n_symmetry_aug - 1) * torch.sum(query_ori[:, 1:self.params.n_symmetry_aug, :], dim=1))
    #             # query_cons = torch.max(query_ori[:, 1:self.params.n_symmetry_aug, :], dim=1)[0]
    #             query_cons = torch.mean(query_ori[:, 1:self.params.n_symmetry_aug, :], dim=1)
    #
    #             query_rec = fusion(torch.cat([query_ori[:,0,:].unsqueeze(1),query_cons.unsqueeze(1)],dim=1)).view(query_ori.shape[0],-1)
    #             if int(self.params.ablation/2) == 1 and self.params.ablation>1:
    #                 # query_rec =  query_cons
    #                 query_rec = query_cons
    #             # query_rec = 0.3*query_ori[:,0,:]+0.7*query_cons
    #
    #         else:
    #             # query_rec =  query_rec[:, 0, :]
    #             query_rec = query_ori[:, 0, :]
    #         # spt_norm = torch.norm(query_rec, p=2, dim=1).unsqueeze(1).expand_as(query_rec)
    #         # query_rec = query_rec.div(spt_norm + 1e-6)
    #         # if no_stym_aug:
    #         #     # print(query_rec.shape)
    #         #     query_rec = (query_rec.unsqueeze(1).expand(query_rec.shape[0],prototype.shape[0],self.dim)+prototype.unsqueeze(0))/2
    #         if no_stym_aug:
    #             query_rec = (query_rec.unsqueeze(1).expand(query_rec.shape[0], prototype.shape[0],
    #                                                        self.dim) + prototype.unsqueeze(0)) / 2
    #         if self.params.LR_rec:
    #
    #             # print(query_rec.shape)
    #             z_query = query_rec.detach().cpu().numpy()
    #             out = torch.from_numpy(clf.predict(z_query))
    #         else:
    #             out = SFC(query_rec)
    #             if no_stym_aug:
    #                 out = out.reshape(query_rec.shape[0],self.params.n_way,-1)
    #                 # out = torch.max(out,dim=1)[0]
    #                 # out = F.softmax(out,dim=-1)
    #                 return out[:,range(self.params.n_way),range(self.params.n_way)]
    #                 # print(out.shape)
    #     return out

    def LR(self,support_z,support_ys,query_z,query_ys):

        clf = LR(penalty='l2',
                 random_state=0,
                 C=self.params.penalty_c,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='multinomial')
        # print(support_z.shape)
        spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        # spt_norm = torch.sqrt(spt_norm )
        spt_normalized = support_z.div(spt_norm  + 1e-6)
        qry_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        # qry_norm = torch.sqrt(qry_norm )
        qry_normalized = query_z.div(qry_norm + 1e-6)
        #
        z_support = spt_normalized.detach().cpu().numpy()
        z_query = qry_normalized.detach().cpu().numpy()

        y_support = np.repeat(range(self.params.n_way), self.n_shot)
        # z_support = support_z.detach().cpu().numpy()
        # y_support = support_ys.view(-1).cpu().numpy()
        # z_query = query_z.detach().cpu().numpy()
        clf.fit(z_support, y_support)

        return torch.from_numpy(clf.predict(z_query))
