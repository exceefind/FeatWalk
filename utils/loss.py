import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def mask_loss(out,gama=0.5):
    # print(out.shape)
    crition = torch.nn.BCELoss()
    out = out.contiguous().view(out.shape[0],-1)
    avg_imp = torch.mean(out,dim=1).unsqueeze(1)
    rate_Sa = torch.mean(torch.where(out >= avg_imp, 1, 0).float(), dim=-1)
    imp_gama = 1 - rate_Sa * gama

    value, ind = torch.sort(out, dim=1, descending=True)
    drop_ind = torch.ceil((1 - imp_gama) * out.shape[-1])
    threshold = value[range(out.shape[0]), drop_ind.long()]
    # threshold = drop_gama * torch.max(att_map.reshape(att_map.shape[0],-1),dim=-1)[0]
    threshold = threshold.unsqueeze(1).expand_as(out)
    fore_mask = torch.where(out >= threshold, 1, 0).float()
    # loss_mask = torch.sum(torch.mean(-fore_mask*torch.log(out+1e-8),1))
    loss_mask = crition(out,fore_mask)
    # print(loss_mask)
    return loss_mask

def uniformity_loss(feat, const_feat,label=None,temp=0.5):
    # sim = (feat * const_feat) / (torch.norm(feat,dim=-1,keepdim=True) * torch.norm(const_feat,dim=-1,keepdim=True))
    sim_aa = torch.cosine_similarity(feat, const_feat, dim=-1)
    # sim_aa = torch.cdist(feat, const_feat, p=2)
    feat_expand = feat.unsqueeze(0).repeat(feat.shape[0],1,1)
    const_feat_expand = const_feat.unsqueeze(1).expand_as(feat_expand)
    sim_ab = torch.cosine_similarity(feat_expand, const_feat_expand,dim=-1)
    # sim_ab = torch.cdist(feat_expand, const_feat_expand, p=2)
    sim_a = torch.exp(sim_aa/temp)
    sim_b = torch.exp(sim_ab/temp)
    sim_tot = torch.sum(sim_b + 1e-6,dim=-1)
    # print(label)
    if label is not None:

        sim_idx = torch.cat([torch.sum(sim_b[i,torch.where(label.squeeze(0) == label.squeeze(0)[i])[0]],dim=-1).unsqueeze(0)
                             for i in range(sim_b.shape[0])],dim=0)
        # print(idx[0,:])
        # print(torch.cat([torch.sum(sim_b[i,idx[i,:]],dim=-1).unsqueeze(0) for i in range(sim_b.shape[0])],dim=0).shape)
        # print(sim_b[0,idx[0,:]].shape)
        # print(sim_b)
        # print( torch.cat([torch.sum(sim_b[i,idx[i,:]],dim=-1).unsqueeze(0) for i in range(sim_b.shape[0])],dim=0))
        p = sim_idx/sim_tot

        # print(torch.sum(sim_b[idx],dim=-1))
    else:
        p = sim_a / sim_tot
    # print(p)
    loss = torch.mean(-torch.log(p+1e-8))
    return loss

def Distance_Correlation(latent, control):
    latent = F.normalize(latent)
    control = F.normalize(control)

    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim=-1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum(torch.square(control.unsqueeze(0) - control.unsqueeze(1)), dim=-1) + 1e-12)

    matrix_A = matrix_a - torch.mean(matrix_a, dim=0, keepdims=True) - torch.mean(matrix_a, dim=1,
                                                                                  keepdims=True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim=0, keepdims=True) - torch.mean(matrix_b, dim=1,
                                                                                  keepdims=True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A) / (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])

    correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    # correlation_r = torch.pow(Gamma_XY,2)/(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r

def area_loss(out,gama=0.5):
    # print(out.shape)
    crition = torch.nn.BCELoss()
    out = out.contiguous().view(out.shape[0],-1)
    y = torch.mean(out,-1)
    avg_imp = torch.mean(out,dim=-1).unsqueeze(1)
    rate_Sa = torch.mean(torch.where(out >= avg_imp, 1, 0).float(), dim=-1)
    imp_gama = rate_Sa * gama
    # print(torch.mean(imp_gama))
    imp_gama = torch.cat([imp_gama.unsqueeze(1),1-imp_gama.unsqueeze(1)],dim=-1)
    y = torch.cat([y.unsqueeze(1), 1 - y.unsqueeze(1)], dim=-1)

    # print(y)
    loss_area = F.kl_div(y.log(),imp_gama, reduction='batchmean')
    # loss_mask = crition(torch.mean(out,dim=-1),imp_gama)
    # print(loss_mask)
    return loss_area

def cosine_sim(out,lab):

    if len(lab.size()) == 1:
        label = torch.zeros((out.size(0),
                             out.size(1))).long().cuda()
        label_range = torch.arange(0, out.size(0)).long()
        label[label_range, lab] = 1
        lab = label

    return torch.mean(torch.abs(out) * lab)

def ce_loss(out, lab,temperature=1,is_softmax = True):

    if is_softmax:
        out = F.softmax(out*temperature, 1)
    if len(lab.size()) == 1:
        label = torch.zeros((out.size(0),
                             out.size(1))).long().cuda()
        label_range = torch.arange(0, out.size(0)).long()
        label[label_range, lab] = 1
        lab = label
    loss = torch.mean(torch.sum(-lab*torch.log(out+1e-8),1))

    return loss

# 计算信息熵的大小
def entropy_loss(out):
    # crition = torch.nn.BCELoss()
    out = F.softmax(out, 1)
    # print(out)
    # pred = torch.ones_like(out)/out.shape[1]
    # loss = crition(pred,out)
    loss = -torch.mean(torch.sum(out*torch.log(out + 1e-8), 1))
    return loss

def Few_loss(out,lab):
    # 目的似乎是实现poly loss，但实践过程中有误
    # 这个损失意义不大
    out = F.softmax(out, 1)
    eps = 2
    n = 1
    poly_head = torch.zeros(out.size(0),out.size(1)).cuda()
    for i in range(n):
        poly_head += eps*1/(i+1)*torch.pow(1-out,(i+1))
    ce_loss = torch.sum(-lab * torch.log(out + 1e-8) - poly_head,1)
    loss = torch.mean(ce_loss)
    return loss

def loc_loss(out_loc,lab):

    out_loc = F.sigmoid(out_loc)
    # print(out_loc)
    log_loc = (-lab) * torch.log(out_loc + 1e-8)-(1-lab)* torch.log(out_loc + 1e-8)
    # loss = torch.mean(torch.sum(log_loc, 1))
    loss = torch.mean(torch.mean(log_loc, 1))

    # out_loc = out_loc.view(out_loc.size(0),out_loc.size(1),-1,2)
    # out_loc = F.softmax(out_loc,dim=3)

    return loss

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    # unsqueeze 在dim维度进行扩展
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_Loss(feat_out,lab,prototypes,epoch,center=False,temperature = 1):
    temperature = 256
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return label_cpu.eq(c).nonzero()[:].squeeze(1)

    feat_cpu = feat_out.cpu()
    label_cpu = lab.cpu()
    prototypes = prototypes.cpu()

    n_classes = prototypes.size(0)
    if len(label_cpu.size()) == 1:
        classes = np.unique(label_cpu)
        #  map :调用函数supp_idsx  classes作为参数列表
        support_idxs = list(map(supp_idxs,classes))
        prototypes_update = torch.stack([feat_cpu[idx_list].mean(0) for idx_list in support_idxs])
    else:
        classes = range(n_classes)
        count = sum(label_cpu, 0)
        # feat_cpu dim  : 64 * 640
        #  label dim    : 64 * 5
        prototypes_update = torch.matmul(feat_cpu.T,label_cpu.float())/torch.tensor(count).float()
        prototypes_update = prototypes_update.T
    # if epoch == 0 :
    # beta = 0.9
    prototypes[classes, :] = prototypes_update.detach()
    # prototypes[classes, :] = beta * prototypes[classes, :] + (1-beta) * prototypes_update.detach()

    if len(lab.size()) == 1:
        label = torch.zeros((feat_cpu.size(0),
                             n_classes)).long().cuda()
        label_range = torch.arange(0, feat_cpu.size(0)).long()
        label[label_range, lab] = 1
        lab = label
    dists = euclidean_dist(feat_cpu,prototypes)/temperature
    # print(dists.shape)
    log_p_y = F.log_softmax(-dists, dim=1)
    y  = F.softmax(-dists,1)

    loss = torch.mean(torch.sum(-lab.cpu() * torch.log(y+1e-8),1))
    # print(loss)
    return loss,prototypes

if __name__ == '__main__':
    # exp = torch.rand((3,5,5))
    # print(area_loss(torch.sigmoid(exp)))
    feat = torch.rand((5, 640,100))
    feat_cons = torch.rand((5,640,100))
    print(Distance_Correlation(feat,feat_cons))
    # print(uniformity_loss(feat,feat_cons))