import tqdm
import torch
import numpy as np
from model import resnet18_gcpl, resnet18_cifar
import matplotlib.pyplot as plt

from data import mnist_data_loader, fashion_mnist_data_loader, cifar10_data_loader, cifar100_data_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_feat():
    # pretrain = 'results_models/gcpl_cifar100_ep_64_sd_2_va_0.0200.pth'
    pretrain = 'saved_models/cifar10/scpn_cifar10_ep_114_ld_2_va_0.8225.pth'
    # pretrain = 'saved_models/mnist/scpn_mnist_ep_97_ld_2_va_0.9939.pth'
    
    
    
    model = resnet18_cifar(
        # num_classes=10, in_channel=1).to(device)
        num_classes=10, in_channel=3).to(device)
        # num_classes=100, in_channel=3).to(device)
    
    model.load_state_dict(torch.load(pretrain, map_location='cpu'))
    model.eval()
    model = model.to(device)

    # c_train, c_test = mnist_data_loader()
    # c_train, c_test = fashion_mnist_data_loader()
    c_train, c_test = cifar10_data_loader()
    # c_train, c_test = cifar100_data_loader()
    

    feats, labels = list(), list()
    for img, label in tqdm.tqdm(list(c_train)):  # list( tup )
    # for img, label in tqdm.tqdm(list(c_test)):  # list( tup )
        
        img = img.float().to(device)
        feat = model(img)
        feats.append(feat.detach().cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0) # axis=0表示沿着数组垂直方向进行拼接


def show(embedding, c):
    plt.figure(figsize=(6, 6))
    # plt.title('Scatter Plot')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    for i in range(10):
        index = np.where(c == i)[0].tolist()
        plt.scatter(embedding[index, 0], embedding[index, 1], s=5)
    plt.legend([str(i) for i in range(10)], loc='upper right')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('scpn_cifar10_ep_114_ld_2_va_0.8225.png', bbox_inches='tight', pad_inches=.05)
    plt.close()
    # plt.show()
 


if __name__ == '__main__':
    feats, labels = get_feat()
    show(feats, labels)
    print('Saved figure!')