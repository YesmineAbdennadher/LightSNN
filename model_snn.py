import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from searchcells.search_cell_snn import Neuronal_Cell
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import gc

def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return  ld

def get_SAR(args,val_loader,network):
  neuron_type = 'LIFNode'
  network=network.eval()

  def nbr_spikes(module, inp, out ,network = network):
     out = out.view(out.size(0), -1)
     batch_num , neuron_num = out.size()
     network.spikes = network.spikes + out.sum().item()
     network.neurons = network.neurons + neuron_num

  for name, module in network.named_modules():
    if neuron_type in str(type(module)):
      module.register_forward_hook(nbr_spikes)


  with torch.no_grad():
    network.neurons = 0
    network.spikes = 0
    data_iterator = iter(val_loader)
    inputs, targets = next(data_iterator)
    inputs= inputs.cuda()
    if args.dataset =='DVS128Gesture':
        inputs = inputs.permute(1, 0, 2, 3, 4)
    outputs = network(inputs)
    spiking_activity = (network.spikes / network.neurons) / args.batch_size

  del network
  del inputs
  del outputs
  torch.cuda.empty_cache()
  return spiking_activity

def get_sahd(args,trainset, network):
        """returns the score k of a network"""
        neuron_type = 'LIFNode'
        search_batchsize = args.search_batch_size
        network=network.eval()
        network.K = np.zeros((search_batchsize, search_batchsize))
        network.num_actfun = 0
        train_data = torch.utils.data.DataLoader(trainset, batch_size=search_batchsize,
                                                   shuffle=True, pin_memory=True, num_workers=4)


        def computing_K_eachtime(module, inp, out,network=network):
            
            if isinstance(out, tuple):
                out = out[1]
            out = out.view(out.size(0), -1)
            batch_num , neuron_num = out.size()
            x = (out > 0).float()


            full_matrix = torch.ones((search_batchsize, search_batchsize)).cuda() * neuron_num
            sparsity = (x.sum(1)/neuron_num).unsqueeze(1)
            norm_K = ((sparsity @ (1-sparsity.t())) + ((1-sparsity) @ sparsity.t())) * neuron_num
            rescale_factor = torch.div(0.5* torch.ones((search_batchsize, search_batchsize)).cuda(), norm_K+1e-3)
            K1_0 = (x @ (1 - x.t()))
            K0_1 = ((1-x) @ x.t())
            K_total = (full_matrix - rescale_factor * (K0_1 + K1_0))

            network.K = network.K + (K_total.cpu().numpy())
            network.num_actfun += 1

        for name, module in network.named_modules():
            if neuron_type in str(type(module)):
                module.register_forward_hook(computing_K_eachtime)

        with torch.no_grad():
            s=[]
            repeat=args.repeat
            for _ in range(repeat):
                #Compute k
                network.K = np.zeros((search_batchsize, search_batchsize))
                network.num_actfun = 0
                data_iterator = iter(train_data)
                inputs, targets = next(data_iterator)
                inputs= inputs.cuda()
                if args.dataset =='DVS128Gesture':
                    inputs = inputs.permute(1, 0, 2, 3, 4)
                outputs = network(inputs)
                s.append(logdet(network.K / (network.num_actfun)))

        del inputs
        del outputs
        torch.cuda.empty_cache()
        return np.mean(s)


def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = torch.cat([(nn.functional.softmax(alpha, 1) > 0.01).float().sum(1) for alpha in arch_parameters], dim=0)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    print('is a single path')
    return True

def prune_func_rank(xargs, arch_parameters, trainset, val_loader,search_space,network, precision=10, prune_number=1):
    INF=1000
    #network_origin = Supernet(xargs,max_nodes=4, search_space=search_space).cuda().train()
    #network_origin.set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    prune_number = min(prune_number, alpha_active[0][0].sum()-1)  # adjust prune_number based on current remaining ops on each edge
    k_all = []  # (k, (edge_idx, op_idx))
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)

    for idx_ct in range(len(arch_parameters)):
        for idx_edge in range(len(arch_parameters[idx_ct])):
            if alpha_active[idx_ct][idx_edge].sum() == 1:
                continue
            for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                    # this edge-op not pruned yet
                    _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                    _arch_param[idx_ct][idx_edge, idx_op] = -INF

                    # ##### get k (score) ########
                    network.set_alphas(_arch_param)
                    k_delta = []
                    repeat = 1

                    for _ in range(repeat):
                        # make sure network_origin and network are identical
                        #for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                         #   param.data.copy_(param_ori.data)
                        #network.set_alphas(_arch_param)

                        k = get_sahd(xargs,trainset, network)
                        k_delta.append(k)
                    k_all.append([np.mean(k_delta), (idx_ct, idx_edge, idx_op)])
                    network.zero_grad()
                    pbar.update(1)
    k_all = sorted(k_all, reverse=True)
    print("N conds:", k_all)
    rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [k_rank]
    for idx, data in enumerate(k_all):
        if idx == 0:
            rankings[data[1]] = [idx]
        else:
            if data[0] == k_all[idx-1][0]:
                # same k as previous
                rankings[data[1]] = [ rankings[k_all[idx-1][1]][0] ]
            else:
                rankings[data[1]] = [ rankings[k_all[idx-1][1]][0] + 1 ]

    rankings_list = [[k, v] for k, v in rankings.items()]# list of [(cell_idx, edge_idx, op_idx), [k_rank]]
    rankings_sum=rankings_list
    edge2choice = {}  # (cell_idx, edge_idx): list of (cell_idx, edge_idx, op_idx) of length prune_number
    for (cell_idx, edge_idx, op_idx), [k_rank] in rankings_sum:
        if (cell_idx, edge_idx) not in edge2choice:
            edge2choice[(cell_idx, edge_idx)] = [(cell_idx, edge_idx, op_idx)]
        elif len(edge2choice[(cell_idx, edge_idx)]) < prune_number:
            edge2choice[(cell_idx, edge_idx)].append((cell_idx, edge_idx, op_idx))
    choices_edges = list(edge2choice.values())


    for choices in choices_edges:
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF
    torch.cuda.empty_cache()
    return arch_parameters, choices_edges



class Supernet(nn.Module):

    def __init__(self, args,max_nodes, search_space, use_stem=True):
        super(Supernet, self).__init__()
        self.args=args
        self.total_timestep = args.timestep
        self.second_avgpooling = args.second_avgpooling
        if self.args.dataset == 'cifar10':
            self.num_class = 10
            self.num_final_neuron = 100
            self.num_cluster = 10
            self.in_channel = 3
            self.img_size = 32
            self.first_out_channel = 128
            self.channel_ratio = 2
            self.spatial_decay = 2 * self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif self.args.dataset == 'cifar100':
            self.num_class = 100
            self.num_final_neuron = 500
            self.num_cluster = 5
            self.in_channel = 3
            self.img_size = 32
            self.channel_ratio = 1
            self.first_out_channel = 128
            self.spatial_decay = 2 *self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1
        elif self.args.dataset == 'DVS128Gesture':
            self.num_class = 11
            self.num_final_neuron = 110
            self.num_cluster = 10
            self.in_channel = 2
            self.img_size = 128
            self.first_out_channel = 128
            self.channel_ratio = 1
            self.spatial_decay = 8 * self.second_avgpooling
            self.classifier_inter_ch = 1024
            self.stem_stride = 1

        self.op_names = deepcopy( search_space )
        self.max_nodes = max_nodes
        self.use_stem = use_stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channel, self.first_out_channel*self.channel_ratio, kernel_size=3, stride=self.stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(self.first_out_channel*self.channel_ratio, affine=True),
        )
        self.max_dvs = nn.MaxPool2d(2,2)

        self.cell1 = Neuronal_Cell(args, self.first_out_channel*self.channel_ratio, self.first_out_channel*self.channel_ratio, self.op_names,self.max_nodes)

        self.downconv1 = nn.Sequential(
            nn.BatchNorm2d(128*self.channel_ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=self.args.threshold, v_reset=0.0, tau=self.args.tau,
                                                      surrogate_function=surrogate.ATan(),
                                                      detach_reset=True),
                                        nn.Conv2d(128*self.channel_ratio, 256*self.channel_ratio, kernel_size=(3, 3),
                                                  stride=(1, 1), padding=(1,1), bias=False),
                                        nn.BatchNorm2d(256*self.channel_ratio, eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True)
                                        )
        self.resdownsample1=nn.MaxPool2d(2,2)

        self.cell2 = Neuronal_Cell(args, 256*self.channel_ratio, 256*self.channel_ratio, self.op_names,self.max_nodes)

        self.last_act = nn.Sequential(
                        nn.BatchNorm2d(256*self.channel_ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        neuron.LIFNode(v_threshold=self.args.threshold, v_reset=0.0, tau=self.args.tau,
                                       surrogate_function=surrogate.ATan(),
                                       detach_reset=True)
        )
        self.resdownsample2=nn.MaxPool2d(self.second_avgpooling,self.second_avgpooling)

        self.classifier = nn.Sequential(
            layer.Dropout(0.5),
            nn.Linear(256*self.channel_ratio*(self.img_size//self.spatial_decay)*(self.img_size//self.spatial_decay), self.classifier_inter_ch, bias=False),
            neuron.LIFNode(v_threshold=self.args.threshold, v_reset=0.0, tau=self.args.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
        nn.Linear(self.classifier_inter_ch, self.num_final_neuron, bias=True))

        self.boost = nn.AvgPool1d(self.num_cluster, self.num_cluster)
        self.arch_parameters = nn.Parameter( 1e-3*torch.randn(self.cell1.num_edges, len(search_space)) )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a =2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_alphas(self):
        return [self.arch_parameters]

    def set_alphas(self, arch_parameters):
        self.arch_parameters.data.copy_(arch_parameters[0].data)

    def neuron_init(self):
        self.cell1.nodes_back=[0]*(self.max_nodes-1)
        self.cell2.nodes_back=[0]*(self.max_nodes-1)

    def forward(self, inputs, return_features=False):
        self.neuron_init()
        acc_voltage = 0
        if self.args.dataset == 'DVS128Gesture':
            batch_size = inputs.size(1)
        else:
            batch_size = inputs.size(0)
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)

        for t in range(self.total_timestep):
            if self.args.dataset == 'DVS128Gesture':
                feature = self.stem(inputs[t])
                feature = self.max_dvs(feature)
            else:
                feature = self.stem(inputs)
            x = self.cell1(feature,alphas)
            x = self.downconv1(x)
            x = self.resdownsample1(x)
            x = self.cell2(x,alphas)
            x = self.last_act(x)
            x = self.resdownsample2(x)
            out = x.view(batch_size, -1)
            x = self.classifier(out)
            acc_voltage = acc_voltage + self.boost(x.unsqueeze(1)).squeeze(1)
        acc_voltage = acc_voltage / self.total_timestep
        return acc_voltage
