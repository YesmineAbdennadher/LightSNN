import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from copy import deepcopy

class ScaleLayer(nn.Module):
   def __init__(self):
       super().__init__()
       self.scale = torch.tensor(0.)

   def forward(self, input):
       return input * self.scale
       
class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class POOLING(nn.Module):

    def __init__(self,args, C_in, C_out):
        super(POOLING, self).__init__()
        self.args=args
        self.op=nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        

    def forward(self, inputs):
        return self.op(inputs)

class LIFConvBN(nn.Module):

    def __init__(self,args, C_in, C_out, kernel_size,  padding):
        super(LIFConvBN, self).__init__()
        self.args=args
        self.op = nn.Sequential(
                      neuron.LIFNode(v_threshold=self.args.threshold, v_reset=0.0, tau=self.args.tau,
                                               surrogate_function=surrogate.ATan(),
                                               detach_reset=True),
                      nn.Conv2d(C_in, C_out, kernel_size=kernel_size,
                                          stride=(1, 1), padding=padding, bias=False),
                    nn.BatchNorm2d(C_out, eps=1e-05, momentum=0.1,
                                               affine=True, track_running_stats=True),
        )


    def forward(self, x):
        return self.op(x)


OPS={'none'        : lambda args,C_in, C_out: ScaleLayer(),
    'pool_3x3': lambda args, C_in, C_out: POOLING(args,C_in, C_out),
    'nor_conv_3x3': lambda  args, C_in, C_out: LIFConvBN(args,C_in, C_out, (3,3), (1,1)),
    'nor_conv_1x1': lambda args, C_in, C_out: LIFConvBN(args,C_in, C_out, (1,1), 0),
    'skip_connect':lambda args, C_in, C_out: Identity()}


class Neuronal_Cell(nn.Module):
    def __init__(self,args, C_in, C_out, op_names,max_nodes):

        super(Neuronal_Cell, self).__init__()
        self.args=args
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out

        if self.args.celltype =='backward':
            for i in range(1, max_nodes):
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                  #if j == 0:
                    xlists = [OPS[op_name](self.args,C_in, C_out) for op_name in op_names]
                    self.edges[node_str] = nn.ModuleList(xlists)
            for i in range(0, self.max_nodes-1):
                for j in range(i+1, self.max_nodes):
                    node_str = '{:}<-{:}'.format(i, j)
                    xlists = [OPS[op_name](self.args,C_in, C_out) for op_name in op_names]
                    self.edges[node_str] = nn.ModuleList(xlists)

            self.edge_keys = sorted(list(self.edges.keys()))
            self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
            self.num_edges = len(self.edges)
            self.nodes_back=[]

        elif self.args.celltype=='forward':
            for i in range(1, max_nodes):
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    xlists = [OPS[op_name](self.args,C_in, C_out) for op_name in op_names]
                    self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)



    def forward(self, inputs, weightss):
        if self.args.celltype=='backward':
            nodes = [inputs]
            for i in range(1, self.max_nodes):
                inter_nodes = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    weights = weightss[self.edge2index[node_str]]
                    inter_nodes.append(sum(layer(nodes[j]+self.nodes_back[j]) * w if w > 0.01 else 0 for layer, w in zip(self.edges[node_str], weights)))  # for pruning purpose
                nodes.append(sum(inter_nodes))
            for i in range(0, self.max_nodes-1):
                inter_nodes = []
                for j in range(i+1, self.max_nodes):
                    node_str = '{:}<-{:}'.format(i, j)
                    weights = weightss[self.edge2index[node_str]]
                    inter_nodes.append(sum(layer(nodes[j]+self.nodes_back[i]) * w if w > 0.01 else 0 for layer, w in zip(self.edges[node_str], weights)))  # for pruning purpose
                self.nodes_back.append(sum(inter_nodes))

        if self.args.celltype=='forward':
            nodes = [inputs]
            for i in range(1, self.max_nodes):
                inter_nodes = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    weights = weightss[self.edge2index[node_str]]
                    inter_nodes.append(sum(layer(nodes[j]) * w if w > 0.01 else 0 for layer, w in zip(self.edges[node_str], weights)))  # for pruning purpose
                nodes.append(sum(inter_nodes))
        return nodes[-1]




