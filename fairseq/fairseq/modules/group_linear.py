from typing import Optional

import torch
from torch import nn

activation_list = [
    'relu', 'leaky', 'selu', 'elu', 'celu', 'prelu', 'sigmoid', 'tanh'
]

def get_activation_layer(name):
    assert name in activation_list, 'Supported activation functions: {}'.format(activation_list)
    if name == 'relu':
        return nn.ReLU(inplace=False)
    elif name == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'celu':
        return nn.CELU(inplace=True)
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(BatchNorm, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps, affine=affine)

    def forward(self, x):
        if x.dim() == 3:
            bsz, seq_len, feature_size = x.size()
            out = self.layer(x.view(-1, feature_size))
            return out.contiguous().view(bsz, seq_len, -1)
        else:
            return self.layer(x)


def get_norm_layer(name, out_features, num_groups=1, eps=1e-5, affine=True):
    assert name in ['gn', 'bn', 'ln'], "Supported normalization functions: ['gn', 'bn', 'ln']"
    if name == 'gn' and num_groups == 1:
        name = 'bn'

    if name == 'bn':
        return BatchNorm(num_features=out_features, eps=eps, affine=affine)
    elif name == 'ln':
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(out_features, eps, affine)
        except:
            return nn.LayerNorm(out_features, eps=eps, elementwise_affine=affine)
    elif name == 'gn':
        return nn.GroupNorm(num_groups=num_groups, num_channels=out_features, eps=eps, affine=affine)


class GroupLinear(nn.Module):
    '''
        This class implements the Grouped Linear Transform
        This is based on the Pyramidal recurrent unit paper:
            https://arxiv.org/abs/1808.09029
    '''

    def __init__(self, in_features: int, out_features: int, n_groups: int = 4, use_bias: bool = False,
                 use_shuffle: bool = False,
                 norm_type: Optional[str] = None, dropout: float = 0.0, act_type: Optional[str] = None):
        '''

        :param in_features: number of input features
        :param out_features: number of output features
        :param n_groups: number of groups in GLT
        :param use_bias: use bias or not
        :param use_shuffle: shuffle features between different groups
        :param norm_type: Normalization type (e.g. LayerNorm)
        :param dropout: Dropout value (default is 0.0)
        :param act_type: Activation type (e.g., Gelu or ReLU)
        '''
        super(GroupLinear, self).__init__()

        assert in_features % n_groups == 0, \
            "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
        assert out_features % n_groups == 0, \
            "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)

        # warning_message = 'Please install custom cuda installation for faster training and inference'

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if use_bias:
            # add 1 in order to make it broadcastable across batch dimension
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        if norm_type is not None:
            self.normalization_fn = get_norm_layer(name=norm_type, out_features=out_groups)
            self.norm_type = norm_type
        else:
            self.normalization_fn = None
            self.norm_type = None

        self.use_dropout = False
        self.drop_p = dropout
        if dropout > 0:
            self.drop_layer = nn.Dropout(p=dropout)
            self.use_dropout = True

        if act_type is not None:
            self.act_fn = get_activation_layer(name=act_type)
            self.act_type = act_type
        else:
            self.act_fn = None
            self.act_type = None

        self.n_groups = n_groups
        self.use_bias = use_bias
        self.shuffle = use_shuffle
        self.feature_shuffle = True if use_shuffle else False

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def process_input_bmm(self, x):
        '''
        N --> Input dimension
        M --> Output dimension
        g --> groups
        G --> gates
        :param x: Input of dimension B x N
        :return: Output of dimension B x M
        '''
        bsz = x.size(0)
        # [B x N] --> [B x g  x N/g]
        x = x.contiguous().view(bsz, self.n_groups, -1)
        # [B x g x N/g] --> [g x B  x N/g]
        x = x.transpose(0, 1)  # transpose so that group is first

        # [g x B  x N/g] x [g x N/g x M/g] --> [g x B x M/g]
        x = torch.bmm(x, self.weights)  # multiply with Weights

        # add bias
        if self.use_bias:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g x B x M/g] --> [B x M/g x g]
            x = x.permute(1, 2, 0)
            # [B x M/g x g] --> [B x g x M/g]
            x = x.contiguous().view(bsz, self.n_groups, -1)
        else:
            # [g x B x M/g] --> [B x g x M/g]
            x = x.transpose(0, 1)  # transpose so that batch is first

        # feature map normalization
        if self.normalization_fn is not None:
            x = self.normalization_fn(x)

        # feature map activation (or thresholding)
        if self.act_fn is not None:
            x = self.act_fn(x)

        return x

    def forward(self, x):
        '''
        :param x: Input of shape [T x B x N] (should work with [B x T x N]
        :return:
        '''
        if x.dim() == 2:
            x = self.process_input_bmm(x)
        elif x.dim() == 3:
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
        else:
            raise NotImplementedError

        # dropout
        if self.use_dropout:
            x = self.drop_layer(x)
        return x

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}, num_groups={n_groups}'
        if self.use_bias:
            s += ', bias={use_bias}'
        if self.shuffle:
            s += ', shuffle={shuffle}'

        if self.norm_type is not None:
            s += ', norm_type={norm_type}'
        if self.act_type is not None:
            s += ', act_type={act_type}'
        if self.drop_p > 0.0:
            s += ', drop_p={drop_p}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        '''
            # of operations in group linear transformation (GLT) are given as:
            Let N and M be dimensions of the input and the output tensor
            Both input and output are split into G groups, so that each input and output group has dimension of N/G and M/G
            Each input group of dimension N/G is mapped to each output group of dimension M/G using a matrix with dimensions [N/G x M/G].
            This mapping involves NM/G^2 additions and NM/G^2 multiplications.
            Since, there are G such groups, we will have total of NM/G addiations and NM/G multipplications.
            Or in simple words, total multiplication-additions (MACs) would be NM/G and FLOPs would be 2NM/G.

            Relationship with # of parameters:
            We have G matrices, each of dimension [N/G x M/G]. The number of parameters in each matrix is NM/G^2.
            Therefore, the total number of parameters in GLT is NM/G.

            MACs = parameters
        '''
        n_mul_wt = self.weights.numel()
        n_add_bias = self.bias.numel() if self.use_bias else 0
        macs = n_mul_wt + n_add_bias
        n_params = n_mul_wt + n_add_bias

        if self.normalization_fn is not None:
            n_params += sum([p.numel() for p in self.normalization_fn.parameters()])
            # MACS are zero for LayerNorm because they can be fused

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }
