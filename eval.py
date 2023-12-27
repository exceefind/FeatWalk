import argparse
import os
import time
import torch

from torch.utils.data import DataLoader
from utils.utils import load_model, set_seed
from data_load.transform_cfg import *
import pprint
from data.datamgr import SetDataManager
from methods.FeatWalk import *

DATA_DIR = 'data'

torch.set_num_threads(4)
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def parse_option():
    parser = argparse.ArgumentParser('arguments for model pre-train')
    # about dataset and network
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs', 'skin198'])
    parser.add_argument('--data_root', type=str, default=DATA_DIR)
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--model', default='resnet12',choices=['resnet12', 'resnet18', 'resnet34', 'conv64'])
    parser.add_argument('--img_size', default=84, type=int, choices=[84,128,160,224])

    # about model :
    parser.add_argument('--drop_gama', default=0.5, type= float)
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument('--drop_rate', default=0.5, type=float)
    parser.add_argument('--reduce_dim', default=128, type=int)
    parser.add_argument('--idea', default='a+-b', choices=['ab', 'a+-b', 'bdc'])
    parser.add_argument('--FPN_list', default=None, nargs='+', type=int)
    parser.add_argument('--flatten_fpn', default=False, action='store_true')

    # about meta test
    parser.add_argument('--val_freq',default=5,type=int)
    parser.add_argument('--set', type=str, default='test', choices=['val', 'test'], help='the set for validation')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--n_aug_support_samples',type=int, default=1)
    parser.add_argument('--n_queries', type=int, default=15)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--num_workers', default=2, type=int)
    #  test_batch_size is 1  mean  1 episode of fsl
    parser.add_argument('--test_batch_size',default=1)

    # setting
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='checkpoint')
    parser.add_argument('--continue_pretrain',default=False,action='store_true')
    parser.add_argument('--test_LR', default=False, action='store_true')
    parser.add_argument('--model_id', default=None, type=str)
    parser.add_argument('--model_type',default='best',choices=['best','last'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--no_save_model', default=False, action='store_true')
    parser.add_argument('--method',default='local_proto',choices=['local_proto','good_metric','stl_deepbdc','confusion','WinSA'])
    parser.add_argument('--distill_model', default=None,type=str,help='about distillation model path')
    parser.add_argument('--penalty_c', default=1.0, type=float)
    parser.add_argument('--idea_variant', default=False, action='store_true')
    parser.add_argument('--test_times', default=1, type=int)

    # confusion representation:
    parser.add_argument('--confusion', default=False, action='store_true')
    parser.add_argument('--n_symmetry_aug', default=1, type=int)
    parser.add_argument('--prompt', default=False, action='store_true')
    parser.add_argument('--constrastive', default=False, action='store_true')
    parser.add_argument('--embeding_way', default='BDC', choices=['BDC','GE','protonet','baseline++'])
    parser.add_argument('--wd_test', type=float, default=5e-4)
    parser.add_argument('--LR', default=False,action='store_true')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--optim', default='Adam',choices=['Adam', 'SGD'])
    parser.add_argument('--my_model', default=False,action='store_true')
    parser.add_argument('--LR_rec', default=False, action='store_true')
    parser.add_argument('--drop_few',default=0.5,type=float)
    parser.add_argument('--skin_split', default=0.5, type=float)
    parser.add_argument('--fix_seed', default=False, action='store_true')
    parser.add_argument('--Loss_ablation', default=3, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--ablation', default=0, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--local_scale', default=0.4 , type=float)
    parser.add_argument('--distill', default=False, action='store_true')
    parser.add_argument('--sfc_bs', default=16, type=int)
    parser.add_argument('--alpha', default=0.5 , type=float)
    parser.add_argument('--sim_temperature', default=64 , type=float)
    parser.add_argument('--measure', default='cosine', choices=['cosine','eudist'])


    args = parser.parse_args()
    if args.deep_emd:
        args.method = 'deep_emd'

    return args


def model_load(args,model):
    # method = 'deep_emd' if args.deep_emd else 'local_match'
    method = args.method
    save_path = os.path.join(args.save_dir, args.dataset + "_" + method + "_resnet12_"+args.model_type
                                            + ("_"+str(args.model_id) if args.model_id else "") + ".pth")
    if args.distill_model is not None:
        save_path = os.path.join(args.save_dir, args.distill_model)
    print('teacher model path: ' + save_path)
    state_dict = torch.load(save_path)['model']
    model.load_state_dict(state_dict)
    return model


def main():
    args = parse_option()
    if args.img_size == 224 and args.transform == 'B':
        args.transform = 'B224'
    if args.img_size == 224 and args.transform == 'B_s':
        args.transform = 'Bs224'

    if args.grid:
        args.n_aug_support_samples = 1
        for i in args.grid:
            args.n_aug_support_samples += i ** 2
        args.n_symmetry_aug = args.n_aug_support_samples
    pprint(args)
    if args.gpu:
        gpu_device = str(args.gpu)
    else:
        gpu_device = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    if args.fix_seed:
        set_seed(args.seed)

    json_file_read = False
    if args.dataset == 'cub':
        novel_file = 'novel.json'
        json_file_read = True
    else:
        novel_file = 'test'
    if args.dataset == 'miniimagenet':
        novel_few_shot_params = dict(n_way=args.n_way, n_support=args.n_shot)
        novel_datamgr = SetDataManager('filelist/miniImageNet_BDC', args.img_size, n_query=args.n_queries,
                                       n_episode=args.n_episodes, json_read=json_file_read,aug_num=args.n_aug_support_samples,args=args,
                                       **novel_few_shot_params)
        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)
        num_classes = 64
    elif args.dataset == 'cub':
        novel_few_shot_params = dict(n_way=args.n_way, n_support=args.n_shot)
        novel_datamgr = SetDataManager('filelist/CUB',args.img_size, n_query=args.n_queries,
                                       n_episode=args.n_episodes, json_read=json_file_read,aug_num=args.n_aug_support_samples,args=args,
                                       **novel_few_shot_params)
        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        novel_few_shot_params = dict(n_way=args.n_way, n_support=args.n_shot)
        novel_datamgr = SetDataManager('filelist/tiered_imagenet', args.img_size, n_query=args.n_queries,args=args,
                                       n_episode=args.n_episodes, json_read=json_file_read,
                                       aug_num=args.n_aug_support_samples,
                                       **novel_few_shot_params)
        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)
        num_classes = 351

    if args.all_mini:
        num_classes = 100

    model = FeatWalk_Net(args,num_classes=num_classes).cuda()
    model = load_model(model,os.path.join(args.save_dir,args.distill_model))

    print("-"*20+"  start meta test...  "+"-"*20)
    acc_sum = 0
    confidence_sum = 0
    for t in range(args.test_times):
        with torch.no_grad():
            tic = time.time()
            mean, confidence = model.meta_test_loop(novel_loader)
            acc_sum += mean
            confidence_sum += confidence
            print()
            print("Time {} :meta_val acc: {:.2f} +- {:.2f}   elapse: {:.2f} min".format(t,mean * 100, confidence * 100,
                                                                               (time.time() - tic) / 60))

    print("{} times \t acc: {:.2f} +- {:.2f}".format(args.test_times, acc_sum/args.test_times * 100, confidence_sum/args.test_times * 100, ))

if __name__ == '__main__':
    main()