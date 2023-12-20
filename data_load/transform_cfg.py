# from __future__ import print_function

import numpy as np
from PIL import Image
import torchvision.transforms as transforms


mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize = transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])



# transform_A = [
#     transforms.Compose([
#         transforms.RandomResizedCrop([image_size,image_size]),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#         transforms.ToTensor(),
#         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
#                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
#     ]),
#
#     transforms.Compose([
#         transforms.Resize([resize_size, resize_size]),
#         transforms.CenterCrop(image_size),
#         # transforms.Resize([84,84]),
#         transforms.ToTensor(),
#         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
#                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
#     ])
# ]
img_size = 84
if img_size == 84:
    resize_size = 92
elif img_size == 224:
    resize_size = 256

transform_A = [
    transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.CenterCrop(img_size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
]

transform_B = [
    transforms.Compose([
            # transforms.Resize([img_size,img_size]),
            # transforms.RandomCrop(img_size, padding=8),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # T.RandomRotation(45),
            transforms.ToTensor(),
            normalize
        ]),

    transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
]

transform_B_s = [
    transforms.Compose([
            # transforms.Resize([resize_size, resize_size]),
            # transforms.CenterCrop(img_size),

            # transforms.RandomCrop(img_size,padding=2),
            # 0214最常用 0.2-0.8
            # transforms.RandomResizedCrop(img_size,scale=(0.2,0.8)),
            transforms.RandomResizedCrop(img_size, scale=(0.3, 0.7)),
            # transforms.RandomResizedCrop(img_size, scale=(0.4, 0.8)),

            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),

    transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
]

img_size = 224
if img_size == 84:
    resize_size = 92
elif img_size == 224:
    resize_size = 256

transform_B224 = [
    transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            # transforms.RandomResizedCrop(img_size, scale=(0.4,0.6)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            normalize
        ]),

    transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
]

transform_Bs224 = [
    transforms.Compose([
            # transforms.RandomResizedCrop(img_size),
            transforms.RandomResizedCrop(img_size, scale=(0.3, 0.7)),

            # transforms.RandomResizedCrop(img_size, scale=(0.4,0.6)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            normalize
        ]),

    transforms.Compose([
        transforms.Resize([resize_size, resize_size]),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
]

transform_B2 = [
    transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        transforms.Resize([92, 92]),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
]

transform_C = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        # transforms.Resize(92, interpolation = PIL.Image.BICUBIC),
        transforms.RandomResizedCrop(80),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Lighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
        # normalize
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(92),
        transforms.CenterCrop(80),
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ])
]

# CIFAR style transformation
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean, std=std)
transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]

transforms_list = ['A', 'B','B_s', 'B2', 'C', 'D']

transforms_options = {
    'A': transform_A,
    'B': transform_B,
    'B224':transform_B224,
    'Bs224':transform_Bs224,
    'B_s':transform_B_s,
    'B2':transform_B2,
    'C': transform_C,
    'D': transform_D,
}
