import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
import numpy as np
import random
from model_snn import Supernet, is_single_path, get_SAR,prune_func_rank
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch.nn.functional as F
import matplotlib.pyplot as plt

def main():
    args = config.get_args()
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join('dataset/', 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=os.path.join('dataset/', 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join('dataset/', 'cifar100'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR100(root=os.path.join('dataset/', 'cifar100'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
    elif args.dataset == 'tinyimagenet':
        trainset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train',
                                        train_transform)
        valset = torchvision.datasets.ImageFolder('tiny-imagenet-200/val',
                                      valid_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True)

    elif args.dataset == 'DVS128Gesture':
      trainset = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')

      test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
      
      train_loader = torch.utils.data.DataLoader(
          dataset=trainset,batch_size=args.batch_size,shuffle=True, drop_last=True,num_workers=4,pin_memory=True)

      val_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=args.batch_size,shuffle=True,drop_last=False,
          num_workers=4,pin_memory=True)

    search_space = args.search_space
    nbr_nodes = args.nbr_nodes
    network = Supernet(args,max_nodes=nbr_nodes,search_space=search_space)

    if args.arch_param == None:
      arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
      for alpha in arch_parameters:
        alpha[:, :] = 0

      network = network.cuda()
      INF = 1000
      start_time = time.time()
      epoch = -1

      while not is_single_path(network):
        epoch += 1
        print('epoch:',epoch)
        arch_parameters, op_pruned = prune_func_rank(args, arch_parameters, trainset,val_loader, search_space,network)
        network.set_alphas(arch_parameters)
      end_time = time.time()

      del network
      torch.cuda.empty_cache()

      print ('-'*7, "best_neuroncell",'-'*7)
      print (arch_parameters)
      print('-' * 30)
      utils.time_record(start_time)
    else:
      ap=[args.arch_param[i:i+5] for i in range(0, len(args.arch_param), 5)]
      arch_parameters=[torch.Tensor(ap)]
      print ('-'*7, "best_neuroncell",'-'*7)
      print (arch_parameters)
      print('-' * 30)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(torch.cuda.memory_summary())

    model = Supernet(args,max_nodes=nbr_nodes, search_space=search_space)
    model=model.cuda()
    model.set_alphas(arch_parameters)
    if args.dataset != 'DVS128Gesture':
        criterion=nn.CrossEntropyLoss()

    if args.savemodel_pth is not None:
        print (torch.load(args.savemodel_pth).keys())
        model.load_state_dict(torch.load(args.savemodel_pth)['state_dict'])
        print ('test only...')
        validate(args, 0, val_loader, model, criterion)
        exit()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    elif args.optimizer == 'adam':
          optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        print ("will be added...")
        exit()

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5),int(args.epochs*0.75)], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= args.learning_rate*0.01)
    else:
        print ("will be added...")
        exit()



    start = time.time()
    val_acc_top=0
    H={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    for epoch in range(1,args.epochs+1):
        print('epoch number:',epoch)
        train_loss,train_acc = train(args, epoch, train_loader, model, criterion,optimizer, scheduler)
        scheduler.step()
        val_loss,val_acc = validate(args, epoch, val_loader, model, criterion)
        print(get_SAR(args, val_loader, model))
        with open('./val_accuracies.txt', 'w') as file:
            file.write(f"{val_acc}\n")
        if val_acc >= val_acc_top:
            val_acc_top=val_acc
            utils.save_checkpoint({'state_dict': model.state_dict(), }, tag=args.exp_name)
        plot(H,args,train_loss,train_acc,val_loss,val_acc)
    utils.time_record(start)

def plot(H,args,train_loss,train_acc,val_loss,val_acc):
  H["train_loss"].append(train_loss)
  H["val_loss"].append(val_loss)
  H["train_acc"].append(train_acc)
  H["val_acc"].append(val_acc)
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(H["train_acc"],label="Train Accuracy")
  plt.plot(H["val_acc"],label="Validation Accuracy")
  plt.title("Accuracy on CIFAR10 dataset")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend(loc="lower left")
  plt.savefig(args.plot_path)

def train(args, epoch, train_data,  model, criterion,optimizer, scheduler):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    if (epoch + 1) % 10 == 0:
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_lr()[0]))

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        if args.dataset =='DVS128Gesture':
            inputs = inputs.permute(1, 0, 2, 3, 4)
            label_onehot = F.one_hot(targets, 11).float()
        outputs = model(inputs)

        if args.dataset=='DVS128Gesture':
            loss = F.mse_loss(outputs, label_onehot)
        else:
            criterion=criterion.cuda()
            loss = criterion(outputs, targets)

        total_loss = loss

        total_loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        if args.dataset=='DVS128Gesture':
            n = inputs.size(1)
        else:
            n = inputs.size(0)

        top1.update(prec1.item(), n)
        train_loss += total_loss.item()
        reset_net(model)
        del inputs, targets, outputs, total_loss
        torch.cuda.empty_cache()
    print('train_loss: %.6f' % (train_loss / len(train_data)), 'train_acc: %.6f' % top1.avg)
    return train_loss/len(train_data) , top1.avg

def validate(args, epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            if args.dataset=='DVS128Gesture':
                inputs = inputs.permute(1, 0, 2, 3, 4)
                label_onehot = F.one_hot(targets, 11).float()

            outputs = model(inputs)

            if args.dataset=='DVS128Gesture':
                loss_v = F.mse_loss(outputs, label_onehot)
            else:
                loss_v = criterion(outputs, targets)

            val_loss += loss_v.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            if args.dataset=='DVS128Gesture':
                n = inputs.size(1)
            else:
                n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            reset_net(model)
            del inputs, targets, outputs, loss_v
            # Optionally, clear the cache
            torch.cuda.empty_cache()
        print('Val_loss: %.6f' % (val_loss/len(val_data)),'[Val_Accuracy epoch:%d] val_acc:%f'% (epoch + 1,val_top1.avg))
        return val_loss/len(val_data) , val_top1.avg

if __name__ == '__main__':
    main()

