import torch
import torch.nn as nn
from CS import OMP, OMP_Separate


def L2_dist(x, y):
    # x : shape [batch, dim]
    # y : shape [num_classes, dim]
    # dist : [batch, num_classes]
    dist = torch.sum(torch.square(x[:, None, :] - y), dim=-1)
    return dist


class DceLoss(nn.Module):
    def __init__(self, t=1):
        super(DceLoss, self).__init__()
        self.t = t
        self.loss = nn.CrossEntropyLoss()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        # print('x.shape, y.shape: ',x.shape, y.shape)
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))  # square 按元素求平方
        return dist

    def forward(self, feats, prototypes, labels):
        logits = -self.L2_dist(feats, prototypes) / self.t
        # print('logits: ', logits.dtype, logits.shape)
        loss = self.loss(logits, labels)
        return loss


class PLoss(nn.Module):
    def __init__(self):
        super(PLoss, self).__init__()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        return dist

    def forward(self, feat, prototype, label):
        dist = self.L2_dist(feat, prototype)
        # gather利用index来索引input特定位置的数值，unsqueeze(-1)拆分元素
        pos_dist = torch.gather(dist, dim=1, index=label.unsqueeze(-1).to(torch.int64))
        pl = torch.mean(pos_dist)
        return pl
    
class CSLoss(nn.Module):
    def __init__(self):
        super(CSLoss, self).__init__()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        return dist

    def forward(self, feat, label, y, A):
        x_r = OMP(y.to(float).cpu().detach().numpy(), A.to(float).cpu().detach().numpy())
        # print('x_r:',x_r)
        # print(feat.shape, x_r.shape)
        dist = self.L2_dist(feat, x_r.cuda())
        cs_dist = torch.gather(dist, dim=1, index=label.cuda().unsqueeze(-1).to(torch.int64))
        # print('cs_dist:',cs_dist)
        cs = torch.mean(cs_dist)
        # print('cs:',cs)
        return cs
    
class Restrict_A(nn.Module):
    def __init__(self):
        super(Restrict_A, self).__init__()

    def forward(self, class_n, A):
        # print('A:',A)
        sparse_n, feature_n = A.shape
        miu_A = torch.abs(torch.matmul(A.T,A))
        ra = (miu_A >= 1/(2*sparse_n-1)).float().sum() / (class_n * feature_n)
        # print('ra:', ra)
        return ra


if __name__ == '__main__':
    features = torch.rand((30, 2))
    prototypes = torch.rand((10, 2))
    labels = torch.rand((30,)).long()
    criterion = DceLoss()
    loss = criterion(features, prototypes, labels)
    print(loss)
