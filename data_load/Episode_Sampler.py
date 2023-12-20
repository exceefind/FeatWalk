import torch
import numpy as np

#  *******************************************************************
#  弃用，主要是因为再batch种无法使得query和support在提取种获得不同的data aug
# ********************************************************************
class EpisodicBatchSampler():

    def __init__(self, label, n_episodes, n_cls, n_shot,n_aug_support_samples=1,):
        # the number of iterations in the dataloader
        self.n_episodes = n_episodes
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_aug_support_samples = n_aug_support_samples

        # all data label
        label = np.array(label)
        # the data index of each class
        self.m_ind = []
        for i in range(max(label) + 1):
            # all data index of this class
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for i in range(self.n_episodes):
            batch = []
            # random sample num_class indexs,e.g. 5
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for cls_i in classes:
                # all data indexs of this class
                samples_cls_i = self.m_ind[cls_i]
                # sample n_per data index of this class
                idx = torch.randperm(len(samples_cls_i))[:self.n_shot]
                # repeat sample n_aug_support_samples times
                for j in range(self.n_aug_support_samples):
                    batch.append(samples_cls_i[idx])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

