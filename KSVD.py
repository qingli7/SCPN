import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def KSVD(Y, D, X, K):
    for k in range(K):
        index = np.nonzero(X[k, :])[0]
        if len(index) == 0:
            continue
        r = (Y - np.dot(D, X))[:, index]
        U, S, V_T = np.linalg.svd(r, full_matrices=False)
        D[:, k] = U[:, 0]
        for j, xj in enumerate(index):
            X[k, xj] = S[0] * V_T[0, j]
    return D, X


def train(Y, Ks, epoches):
    np.random.seed(1024)
    images = []
    for K in Ks:
        # 初始化D，从Y中随机选取K列作为D
        U, _, _ = np.linalg.svd(Y)
        D = U[:, :K]
        for epoch in range(epoches):
            # 每一次更新D之后由OMP算法求得稀疏矩阵X
            X = linear_model.orthogonal_mp((D-D.min())/(D.max()-D.min()), Y)
            # KSVD算法更新D
            D, X = KSVD(Y, D, X, K)
            # 计算损失并输出
            L2_loss = (((Y - np.dot(D, X)) ** 2) ** 0.2).mean()
            print('K: {} | Epoch: {} | L2 loss: {}'.format(K, epoch, L2_loss))
        # 最后一轮更新D之后还需要拟合一下新的X
        print('D:', D.shape)
        X = linear_model.orthogonal_mp((D-D.min())/(D.max()-D.min()), Y)
        print('X:', X.shape)
        # 重构图片
        rebuilded_image = np.clip(np.dot((D-D.min())/(D.max()-D.min()), X).reshape(*image.shape), 0, 1)
        print('rebuilded_image:',rebuilded_image.shape)
        images.append(rebuilded_image)
        print('')
    return images


if __name__ == '__main__':
    image = mpimg.imread('data/sheep.png')
    print('Image shape: {}'.format(image.shape))
    plt.imshow(image)
    Y = image.reshape(image.shape[0], -1)
    print("Y shape: {}".format(Y.shape))
    
    
    Ks = [20, 50, 100, 300]
    images = train(Y, Ks, epoches=3)

    _, axarr = plt.subplots(1, len(Ks)+1, figsize=((len(Ks)+1)*5, 5))
    axarr[0].imshow(image)
    axarr[0].set_title('Original image')
    for i, img in enumerate(images):
        axarr[i+1].imshow(img)
        axarr[i+1].set_title('K={}'.format(Ks[i]))
    plt.savefig('KSVD.png', bbox_inches='tight', pad_inches=.05)
    