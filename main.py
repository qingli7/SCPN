import torch
import argparse
import shutil
from model import resnet18_gcpl, resnet18_cifar
from loss import DceLoss, PLoss, CSLoss, Restrict_A
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data import mnist_data_loader, fashion_mnist_data_loader, cifar10_data_loader, cifar100_data_loader

parser = argparse.ArgumentParser(description='Train Convolutionary Prototype Learning Models')

parser.add_argument('--epochs', default=301, type=int, help='total number of epochs to run')

# parser.add_argument('--data_name', default='mnist', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=1, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='fashion_mnist', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=1, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

parser.add_argument('--data_name', default='cifar10', type=str, help='dataset name to use')
parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='cifar100', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=100, type=int, help='class number for the dataset')

parser.add_argument('--feature_dim', default=512, type=int, help='feature dimension of original data')
parser.add_argument('--latent_dim', default=2, type=int, help='latent dimension of prototype feature') # 2, 32, 64, 128
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--pl_weight', type=float, default=0.001, help='pl learning weight')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    print(args)
    # train_loader, val_loader = mnist_data_loader(batch_size=args.batch_size)
    # train_loader, val_loader = fashion_mnist_data_loader(batch_size=args.batch_size)
    train_loader, val_loader = cifar10_data_loader(batch_size=args.batch_size)
    # train_loader, val_loader = cifar100_data_loader(batch_size=args.batch_size)
    
    # model = resnet18_gcpl(num_classes=args.num_classes, in_channel=args.data_channel,latent_dim=args.latent_dim).to(device)
    model = resnet18_cifar(num_classes=args.num_classes, in_channel=args.data_channel,latent_dim=args.latent_dim).to(device)
    
    dce = DceLoss()
    pl = PLoss()
    cs = CSLoss()
    ra = Restrict_A()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)  #
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 210, 270], gamma=0.1)
    
    temp_acc = 0.8
    for epoch in range(args.epochs):
        train_loss, train_correct, train_total = 0, 0, 0
        val_loss, val_correct, val_total = 0, 0, 0

        model.train()
        # for inputs, targets in tqdm(train_loader):
        for _, (inputs, targets) in enumerate(train_loader):  # tqdm
            inputs = inputs.float().to(device)
            # print('inputs: ', inputs.shape)
            targets = targets.long().to(device)
            feat = model(inputs)
            # print('feat: ', feat.shape, torch.isnan(feat))
            ofeat = torch.mm(feat, model.A.T)
            optimizer.zero_grad()
            # print('model.prototypes.shape: ',model.prototypes.shape)
            loss = dce(ofeat, model.prototypes, targets) + args.pl_weight * pl(ofeat, model.prototypes, targets)\
                + 0.001 * cs(feat, targets, model.prototypes, model.A) + 0.001 * ra(args.num_classes, model.A)
            # print('dce:',dce(ofeat, model.prototypes, targets),'pl:',pl(ofeat, model.prototypes, targets),'cs:',pl(ofeat, model.prototypes, targets),'ra:',0.001 * ra(args.num_classes, model.A))
            # loss1 =  dce(ofeat, model.prototypes, targets)
            # loss2 = 0.001 * cs(feat, targets, model.prototypes, model.A)
            # loss3 =  0.001 * ra(args.num_classes, model.A)
            # print('dce:', loss1 ,'cs:', loss2, 'reA:', loss3)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            _, predicted = torch.min(dce.L2_dist(ofeat, model.prototypes), dim=1)
            train_total += targets.size(0)  # batch_size: 64
            train_correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / len(train_loader)  # / 938 = 60032 / 64
        train_acc = train_correct / train_total  # / 60,000

        scheduler.step()

        model.eval()
        with torch.no_grad():
            # for inputs, targets in tqdm(val_loader):
            for _, (inputs, targets) in enumerate(val_loader):   # tqdm
                inputs = inputs.float().to(device)
                targets = targets.long().to(device)
                feat = model(inputs)
                ofeat = torch.mm(feat, model.A.T)
                loss = dce(ofeat, model.prototypes, targets) + args.pl_weight * pl(ofeat, model.prototypes, targets)\
                    + 0.001 * cs(feat, targets, model.prototypes, model.A) + 0.001 * ra(args.num_classes, model.A)
                val_loss = val_loss + loss.item()
                _, predicted = torch.min(dce.L2_dist(ofeat, model.prototypes), dim=1)
                val_total += inputs.shape[0]
                val_correct += predicted.eq(targets).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

        if (epoch % 10 == 0) or (val_acc >= temp_acc):
            temp_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/%s/scpn_%s_ep_%d_ld_%d_va_%.4f.pth'
                       % (args.data_name, args.data_name, epoch, args.latent_dim, val_acc))

        print(
            'Epoch : %03d  Train Loss: %.3f | Train Acc: %.3f%% | Val Loss: %.3f | Val Acc: %.3f%%'
            % (epoch, train_loss, 100 * train_acc, val_loss, 100 * val_acc))

        # record training logs
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)


if __name__ == '__main__':
    # epochs = 100
    # batch_size = 64
    # num_classes = 10

    # remove old log file
    logdir = './tensorboard/GCPL/'
    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir)

    # main(epochs, batch_size, lr=1e-3, num_classes=10, in_channel=1)
    
    main()
