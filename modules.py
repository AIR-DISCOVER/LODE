
import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import MinkowskiEngine as ME
import torch.nn.functional as F

from IPython import embed


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=3,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, config={}):
        super().__init__()
        self.mode = mode
        self.config = config
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        # print(self)

    def forward(self, shapes, coords, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = coords.clone().detach().requires_grad_(True)
        net_input_org = torch.cat([shapes, coords_org], dim=-1)

        if self.config['TRAIN']['encode_xyz'] == True:
            net_input = torch.cat([net_input_org[..., :self.config['TRAIN']['shape_embedding_size']], \
                            encode_position(net_input_org[..., self.config['TRAIN']['shape_embedding_size']:], \
                                self.config['TRAIN']['encode_levels'], self.config['TRAIN']['inc_input'])], dim=-1)
        else:
            net_input = net_input_org

        output = self.net(net_input, get_subdict(params, 'net'))
        return {'model_in': net_input_org, 'model_out': output}


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def encode_position(input, levels, inc_input):
    '''
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.

    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    '''

    # this is already doing 'log_sampling' in the official code.
    result_list = [input] if inc_input else []
    for i in range(levels):
        temp = 2.0**i * input  # (..., C)
        result_list.append(torch.sin(temp))  # (..., C)
        result_list.append(torch.cos(temp))  # (..., C)

    result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))

################### sc ###################
class D_Net_shape(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512]

    def __init__(self, config={}):
        nn.Module.__init__(self)

        chunk_size = config['TRAIN']['chunk_size']
        shape_embedding_size = config['TRAIN']['shape_embedding_size']

        self.pruning_choice = config['TRAIN']['D_TRAIN']['pruning_choice']

        if config['TRAIN']['D_TRAIN']['nonlinearity'] == 'elu':
            nl = ME.MinkowskiELU()
        elif config['TRAIN']['D_TRAIN']['nonlinearity'] == 'relu':
            nl = ME.MinkowskiReLU()

        if config['TRAIN']['D_TRAIN']['D_input'] == 'radial_height':
            in_channels = 2
        else:
            in_channels = 1
            
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            nl,
        )

        self.enc_res_block_s1_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            nl,
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            nl,
        )

        self.enc_res_block_s1_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            nl,
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            nl,
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            nl,
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            nl,
        )

        self.enc_res_block_s2_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            nl,
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            nl,
        )

        self.enc_res_block_s2_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            nl,
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            nl,
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            nl,
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            nl,
        )

        self.enc_res_block_s4_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            nl,
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            nl,
        )

        self.enc_res_block_s4_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            nl,
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            nl,
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            nl,
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            nl,
        )

        self.enc_res_block_s8_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            nl,
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            nl,
        )

        self.enc_res_block_s8_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            nl,
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            nl,
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            nl,
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            nl,
        )

        self.enc_res_block_s16_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            nl,
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            nl,
        )

        self.enc_res_block_s16_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            nl,
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            nl,
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            nl,
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            nl,
        )

        self.enc_res_block_s32_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            nl,
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            nl,
        )

        self.enc_res_block_s32_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            nl,
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            nl,
        )

        # Decoder
        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            nl,
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            nl,
        )

        self.dec_res_block_s16_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            nl,
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            nl,
        )

        self.dec_res_block_s16_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            nl,
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            nl,
        )

        if self.pruning_choice[0]:
            self.dec_s16_cls = ME.MinkowskiConvolution(
                dec_ch[4], 1, kernel_size=1, bias=True, dimension=3
            )

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            nl,
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            nl,
        )

        self.dec_res_block_s8_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            nl,
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            nl,
        )

        self.dec_res_block_s8_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            nl,
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            nl,
        )
        
        if self.pruning_choice[1]:
            self.dec_s8_cls = ME.MinkowskiConvolution(
                dec_ch[3], 1, kernel_size=1, bias=True, dimension=3
            )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            nl,
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            nl,
        )

        self.dec_res_block_s4_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            nl,
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            nl,
        )

        self.dec_res_block_s4_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            nl,
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            nl,
        )

        if self.pruning_choice[2]:
            self.dec_s4_cls = ME.MinkowskiConvolution(
                dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
            )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            nl,
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            nl,
        )

        self.dec_res_block_s2_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            nl,
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            nl,
        )

        self.dec_res_block_s2_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            nl,
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            nl,
        )

        if self.pruning_choice[3]:
            self.dec_s2_cls = ME.MinkowskiConvolution(
                dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
            )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            nl,
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            nl,
        )

        self.dec_res_block_s1_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            nl,
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            nl,
        )

        self.dec_res_block_s1_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            nl,
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            nl,
        )

        if self.pruning_choice[4]:
            self.dec_s1_cls = ME.MinkowskiConvolution(
                dec_ch[0], 1, kernel_size=1, bias=True, dimension=3
            )

        # pruning
        self.pruning = ME.MinkowskiPruning()

        if config['TRAIN']['D_TRAIN']['output_layers'] == 2:
            self.se_out = nn.Sequential(
                ME.MinkowskiConvolution(dec_ch[0], shape_embedding_size // 2, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(shape_embedding_size // 2),
                ME.MinkowskiLeakyReLU(),
                
                ME.MinkowskiConvolution(shape_embedding_size // 2, shape_embedding_size, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(shape_embedding_size),
                ME.MinkowskiLeakyReLU(),
                
                ME.MinkowskiAvgPooling(kernel_size=chunk_size, stride=chunk_size, dimension=3),
            )
        elif config['TRAIN']['D_TRAIN']['output_layers'] == 4:
            self.se_out = nn.Sequential(
                ME.MinkowskiConvolution(dec_ch[0], shape_embedding_size // 8, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(shape_embedding_size // 8),
                ME.MinkowskiLeakyReLU(),
                
                ME.MinkowskiConvolution(shape_embedding_size // 8, shape_embedding_size // 4, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(shape_embedding_size // 4),
                ME.MinkowskiLeakyReLU(),
                
                ME.MinkowskiConvolution(shape_embedding_size // 4, shape_embedding_size // 2, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(shape_embedding_size // 2),
                ME.MinkowskiLeakyReLU(),
                
                ME.MinkowskiConvolution(shape_embedding_size // 2, shape_embedding_size, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(shape_embedding_size),
                ME.MinkowskiLeakyReLU(),
                
                ME.MinkowskiAvgPooling(kernel_size=chunk_size, stride=chunk_size, dimension=3),
            )

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def forward(self, partial_in, target_key):
        out_cls, targets = [], []

        enc_s1 = self.enc_block_s1(partial_in)
        enc_res_s1 = self.enc_res_block_s1_0(enc_s1)
        enc_s1 = enc_s1 + enc_res_s1
        enc_res_s1 = self.enc_res_block_s1_1(enc_s1)
        enc_s1 = enc_s1 + enc_res_s1

        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_res_s2 = self.enc_res_block_s2_0(enc_s2)
        enc_s2 = enc_s2 + enc_res_s2
        enc_res_s2 = self.enc_res_block_s2_1(enc_s2)
        enc_s2 = enc_s2 + enc_res_s2

        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_res_s4 = self.enc_res_block_s4_0(enc_s4)
        enc_s4 = enc_s4 + enc_res_s4
        enc_res_s4 = self.enc_res_block_s4_1(enc_s4)
        enc_s4 = enc_s4 + enc_res_s4

        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_res_s8 = self.enc_res_block_s8_0(enc_s8)
        enc_s8 = enc_s8 + enc_res_s8
        enc_res_s8 = self.enc_res_block_s8_1(enc_s8)
        enc_s8 = enc_s8 + enc_res_s8

        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_res_s16 = self.enc_res_block_s16_0(enc_s16)
        enc_s16 = enc_s16 + enc_res_s16
        enc_res_s16 = self.enc_res_block_s16_1(enc_s16)
        enc_s16 = enc_s16 + enc_res_s16

        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_res_s32 = self.enc_res_block_s32_0(enc_s32)
        enc_s32 = enc_s32 + enc_res_s32
        enc_res_s32 = self.enc_res_block_s32_1(enc_s32)
        enc_s32 = enc_s32 + enc_res_s32

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(enc_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_res_s16 = self.dec_res_block_s16_0(dec_s16)
        dec_s16 = dec_s16 + dec_res_s16
        dec_res_s16 = self.dec_res_block_s16_1(dec_s16)
        dec_s16 = dec_s16 + dec_res_s16

        if self.pruning_choice[0]:
            dec_s16_cls = self.dec_s16_cls(dec_s16)

            target = self.get_target(dec_s16, target_key)
            targets.append(target)
            out_cls.append(dec_s16_cls)
            keep_s16 = (dec_s16_cls.F > 0).squeeze()

            if self.training:
                keep_s16 += target

            # Remove voxels s16
            if keep_s16.sum() > 0:
                dec_s16 = self.pruning(dec_s16, keep_s16)            

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_res_s8 = self.dec_res_block_s8_0(dec_s8)
        dec_s8 = dec_s8 + dec_res_s8
        dec_res_s8 = self.dec_res_block_s8_1(dec_s8)
        dec_s8 = dec_s8 + dec_res_s8

        if self.pruning_choice[1]:
            dec_s8_cls = self.dec_s8_cls(dec_s8)

            target = self.get_target(dec_s8, target_key)
            targets.append(target)
            out_cls.append(dec_s8_cls)
            keep_s8 = (dec_s8_cls.F > 0).squeeze()

            if self.training:
                keep_s8 += target

            # Remove voxels s16
            if keep_s8.sum() > 0:
                dec_s8 = self.pruning(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_res_s4 = self.dec_res_block_s4_0(dec_s4)
        dec_s4 = dec_s4 + dec_res_s4
        dec_res_s4 = self.dec_res_block_s4_1(dec_s4)
        dec_s4 = dec_s4 + dec_res_s4

        if self.pruning_choice[2]:
            dec_s4_cls = self.dec_s4_cls(dec_s4)

            target = self.get_target(dec_s4, target_key)
            targets.append(target)
            out_cls.append(dec_s4_cls)
            keep_s4 = (dec_s4_cls.F > 0).squeeze()

            if self.training:
                keep_s4 += target

            # Remove voxels s4
            if keep_s4.sum() > 0:
                dec_s4 = self.pruning(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_res_s2 = self.dec_res_block_s2_0(dec_s2)
        dec_s2 = dec_s2 + dec_res_s2
        dec_res_s2 = self.dec_res_block_s2_1(dec_s2)
        dec_s2 = dec_s2 + dec_res_s2

        if self.pruning_choice[3]:
            dec_s2_cls = self.dec_s2_cls(dec_s2)

            target = self.get_target(dec_s2, target_key)
            targets.append(target)
            out_cls.append(dec_s2_cls)
            keep_s2 = (dec_s2_cls.F > 0).squeeze()

            if self.training:
                keep_s2 += target

            # Remove voxels s2
            if keep_s2.sum() > 0:
                dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_res_s1 = self.dec_res_block_s1_0(dec_s1)
        dec_s1 = dec_s1 + dec_res_s1
        dec_res_s1 = self.dec_res_block_s1_1(dec_s1)
        dec_s1 = dec_s1 + dec_res_s1

        if self.pruning_choice[4]:
            dec_s1_cls = self.dec_s1_cls(dec_s1)

            target = self.get_target(dec_s1, target_key)
            targets.append(target)
            out_cls.append(dec_s1_cls)
            keep_s1 = (dec_s1_cls.F > 0).squeeze()

            # Last layer does not require adding the target
            # if self.training:
            #     keep_s1 += target

            # Remove voxels s1
            if keep_s1.sum() > 0:
                dec_s1 = self.pruning(dec_s1, keep_s1)

        shape_out = self.se_out(dec_s1)

        return out_cls, targets, dec_s1, shape_out


class D_Net(nn.Module):
    def __init__(self, config={}):
        nn.Module.__init__(self)
        self.config = config
        self.shapes = D_Net_shape(config)

    def forward(self, partial_in, target_key):
        out_cls, targets, sout, shape_out = self.shapes(partial_in, target_key)
        return out_cls, targets, sout, shape_out

################### sc ###################


class Seg_Net(nn.Module):
    ENC_CHANNELS = [32, 64, 128, 256, 512]
    DEC_CHANNELS = [32, 64, 128, 256, 512]

    def __init__(self, config={}, in_channels=1, out_channels=20):
        nn.Module.__init__(self)

        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s1_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s1_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s2_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s2_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s4_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s4_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s8_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s8_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s16_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s16_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        # Decoder
        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s8_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s8_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s4_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s4_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s2_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s2_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s1_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s1_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.out_block = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
        )

    def forward(self, partial_in):
        enc_s1 = self.enc_block_s1(partial_in)
        enc_res_s1 = self.enc_res_block_s1_0(enc_s1)
        enc_s1 = enc_s1 + enc_res_s1
        enc_res_s1 = self.enc_res_block_s1_1(enc_s1)
        enc_s1 = enc_s1 + enc_res_s1

        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_res_s2 = self.enc_res_block_s2_0(enc_s2)
        enc_s2 = enc_s2 + enc_res_s2
        enc_res_s2 = self.enc_res_block_s2_1(enc_s2)
        enc_s2 = enc_s2 + enc_res_s2

        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_res_s4 = self.enc_res_block_s4_0(enc_s4)
        enc_s4 = enc_s4 + enc_res_s4
        enc_res_s4 = self.enc_res_block_s4_1(enc_s4)
        enc_s4 = enc_s4 + enc_res_s4

        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_res_s8 = self.enc_res_block_s8_0(enc_s8)
        enc_s8 = enc_s8 + enc_res_s8
        enc_res_s8 = self.enc_res_block_s8_1(enc_s8)
        enc_s8 = enc_s8 + enc_res_s8

        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_res_s16 = self.enc_res_block_s16_0(enc_s16)
        enc_s16 = enc_s16 + enc_res_s16
        enc_res_s16 = self.enc_res_block_s16_1(enc_s16)
        enc_s16 = enc_s16 + enc_res_s16

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(enc_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_res_s8 = self.dec_res_block_s8_0(dec_s8)
        dec_s8 = dec_s8 + dec_res_s8
        dec_res_s8 = self.dec_res_block_s8_1(dec_s8)
        dec_s8 = dec_s8 + dec_res_s8

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_res_s4 = self.dec_res_block_s4_0(dec_s4)
        dec_s4 = dec_s4 + dec_res_s4
        dec_res_s4 = self.dec_res_block_s4_1(dec_s4)
        dec_s4 = dec_s4 + dec_res_s4

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_res_s2 = self.dec_res_block_s2_0(dec_s2)
        dec_s2 = dec_s2 + dec_res_s2
        dec_res_s2 = self.dec_res_block_s2_1(dec_s2)
        dec_s2 = dec_s2 + dec_res_s2

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_res_s1 = self.dec_res_block_s1_0(dec_s1)
        dec_s1 = dec_s1 + dec_res_s1
        dec_res_s1 = self.dec_res_block_s1_1(dec_s1)
        dec_s1 = dec_s1 + dec_res_s1

        out_class = self.out_block(dec_s1)

        return out_class


class SSCNet_Decoder(nn.Module):
    def __init__(self, in_channels, nPlanes, classes):
        super().__init__()
        # Block 1
        self.b1_conv1_0=nn.Sequential(nn.Conv3d(in_channels, 16, 3, 1, padding=1), nn.BatchNorm3d(16),nn.ReLU())
        self.b1_conv1_1=nn.Sequential(nn.Conv3d(16, 16, 3, 1, padding=1), nn.BatchNorm3d(16),nn.ReLU())
        self.b1_conv1_2=nn.Sequential(nn.Conv3d(16, 16, 3, 2, padding=1), nn.BatchNorm3d(16),nn.ReLU())
        
        self.b1_conv2=nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]),nn.ReLU())
        self.b1_conv3=nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]),nn.ReLU())
        self.b1_res=nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1,padding=1), nn.BatchNorm3d(nPlanes[0]),nn.ReLU())

        # Block 2
        self.b2_conv1=nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_conv2=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_res=nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())

        # Block 2_1
        self.b2_1_conv1=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_1_conv2=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_1_res=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())

        # Block 2_2
        self.b2_2_conv1=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_2_conv2=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_2_res=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())

        # Block 3
        self.b3_conv1=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),nn.ReLU())
        self.b3_conv2=nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),nn.ReLU())

        # Block 4
        self.b4_conv1=nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[3], 3, 1, dilation=2, padding=2), nn.BatchNorm3d(nPlanes[3]),nn.ReLU())
        self.b4_conv2=nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[3], 3, 1, dilation=2, padding=2), nn.BatchNorm3d(nPlanes[3]),nn.ReLU())

        # Block 5
        self.b5_conv1=nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[4], 3, 1, dilation=4, padding=4), nn.BatchNorm3d(nPlanes[4]),nn.ReLU())
        self.b5_conv2=nn.Sequential(nn.Conv3d(nPlanes[4], nPlanes[4], 3, 1, dilation=4, padding=4), nn.BatchNorm3d(nPlanes[4]),nn.ReLU())
        
        # Block 6
        self.b6_conv1=nn.Sequential(nn.Conv3d(nPlanes[4], nPlanes[5], 3, 1, dilation=8, padding=8), nn.BatchNorm3d(nPlanes[5]),nn.ReLU())
        self.b6_conv2=nn.Sequential(nn.Conv3d(nPlanes[5], nPlanes[5], 3, 1, dilation=8, padding=8), nn.BatchNorm3d(nPlanes[5]),nn.ReLU())
        
        # Prediction
        self.pre_conv1=nn.Sequential(nn.Conv3d(nPlanes[2]+nPlanes[3]+nPlanes[4]+nPlanes[5], int((nPlanes[2]+nPlanes[3]+nPlanes[4]+nPlanes[5])/4*2), 1, 1),\
                                     nn.BatchNorm3d(int((nPlanes[2]+nPlanes[3]+nPlanes[4]+nPlanes[5])/4*2)),nn.ReLU())
        self.pre_conv2=nn.Sequential(nn.Conv3d(int((nPlanes[2]+nPlanes[3]+nPlanes[4]+nPlanes[5])/4*2), classes, 1, 1))

    def forward(self, x):
        # Block 1
        x = self.b1_conv1_0(x)
        x = self.b1_conv1_1(x)
        x = self.b1_conv1_2(x)

        # Block 1_1
        res_x = self.b1_res(x)
        x = self.b1_conv2(x)
        x = self.b1_conv3(x)
        x = x + res_x

        # Block 2
        res_x = self.b2_res(x)
        x = self.b2_conv1(x)
        x = self.b2_conv2(x)
        x = x + res_x

        # Block 2_1
        res_x = self.b2_1_res(x)
        x = self.b2_1_conv1(x)
        x = self.b2_1_conv2(x)
        x = x + res_x

        # Block 2_2
        res_x = self.b2_2_res(x)
        x = self.b2_2_conv1(x)
        x = self.b2_2_conv2(x)
        x = x + res_x

        # Block 3
        b3_x1 = self.b3_conv1(x)
        b3_x2 = self.b3_conv2(b3_x1)
        b3_x = b3_x1 + b3_x2

        # Block 4
        b4_x1 = self.b4_conv1(b3_x)
        b4_x2 = self.b4_conv2(b4_x1)
        b4_x = b4_x1 + b4_x2

        # Block 5
        b5_x1 = self.b5_conv1(b4_x)
        b5_x2 = self.b5_conv2(b5_x1)
        b5_x = b5_x1 + b5_x2

        # Block 6
        b6_x1 = self.b6_conv1(b5_x)
        b6_x2 = self.b6_conv2(b6_x1)
        b6_x = b6_x1 + b6_x2

        # Concat b3,b4,b5
        x = torch.cat((b3_x, b4_x, b5_x, b6_x),dim=1)

        # Prediction
        x = self.pre_conv1(x)
        x = self.pre_conv2(x)
        return x


class PixelShuffle3D(nn.Module):
    '''
    3D pixelShuffle
    '''
    def __init__(self, upscale_factor):
        '''
        :param upscale_factor: int
        '''
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class SSC_Net(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        classes = config['TRAIN']['class_count']

        in_channels = config['TRAIN']['class_count']
        
        m = 32

        self.Decoder = SSCNet_Decoder(in_channels=in_channels, nPlanes=[m, m, m, m, m, m], classes=classes)
        self.upsample = nn.Sequential(nn.Conv3d(in_channels=classes, out_channels=classes * 8, kernel_size=1, stride=1),
                                      nn.BatchNorm3d(classes * 8), nn.ReLU(), PixelShuffle3D(upscale_factor=2))

    def forward(self, x_in):
        x = self.Decoder(x_in)
        x = self.upsample(x)

        return x


class Com_Net(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512]

    def __init__(self, config={}, in_channels=20):
        nn.Module.__init__(self)

        self.config = config
        chunk_size = config['TRAIN']['chunk_size']
        shape_embedding_size = config['TRAIN']['shape_embedding_size']

        self.pruning_choice = config['TRAIN']['D_TRAIN']['pruning_choice']

        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s1_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s1_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s2_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s2_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s4_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s4_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s8_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s8_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s16_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s16_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s32_0 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
        )

        self.enc_res_block_s32_1 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
        )

        # Decoder
        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s16_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s16_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
        )


        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s8_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s8_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s4_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s4_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s2_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s2_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s1_0 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.dec_res_block_s1_1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

        self.se_out = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], shape_embedding_size // 2, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(shape_embedding_size // 2),
            ME.MinkowskiLeakyReLU(),
            
            ME.MinkowskiConvolution(shape_embedding_size // 2, shape_embedding_size, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(shape_embedding_size),
            ME.MinkowskiLeakyReLU(),
            
            ME.MinkowskiAvgPooling(kernel_size=chunk_size, stride=chunk_size, dimension=3),
        )

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def forward(self, partial_in, target_key):
        out_cls, targets = [], []

        enc_s1 = self.enc_block_s1(partial_in)
        enc_res_s1 = self.enc_res_block_s1_0(enc_s1)
        enc_s1 = enc_s1 + enc_res_s1
        enc_res_s1 = self.enc_res_block_s1_1(enc_s1)
        enc_s1 = enc_s1 + enc_res_s1

        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_res_s2 = self.enc_res_block_s2_0(enc_s2)
        enc_s2 = enc_s2 + enc_res_s2
        enc_res_s2 = self.enc_res_block_s2_1(enc_s2)
        enc_s2 = enc_s2 + enc_res_s2

        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_res_s4 = self.enc_res_block_s4_0(enc_s4)
        enc_s4 = enc_s4 + enc_res_s4
        enc_res_s4 = self.enc_res_block_s4_1(enc_s4)
        enc_s4 = enc_s4 + enc_res_s4

        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_res_s8 = self.enc_res_block_s8_0(enc_s8)
        enc_s8 = enc_s8 + enc_res_s8
        enc_res_s8 = self.enc_res_block_s8_1(enc_s8)
        enc_s8 = enc_s8 + enc_res_s8

        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_res_s16 = self.enc_res_block_s16_0(enc_s16)
        enc_s16 = enc_s16 + enc_res_s16
        enc_res_s16 = self.enc_res_block_s16_1(enc_s16)
        enc_s16 = enc_s16 + enc_res_s16

        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_res_s32 = self.enc_res_block_s32_0(enc_s32)
        enc_s32 = enc_s32 + enc_res_s32
        enc_res_s32 = self.enc_res_block_s32_1(enc_s32)
        enc_s32 = enc_s32 + enc_res_s32

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(enc_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_res_s16 = self.dec_res_block_s16_0(dec_s16)
        dec_s16 = dec_s16 + dec_res_s16
        dec_res_s16 = self.dec_res_block_s16_1(dec_s16)
        dec_s16 = dec_s16 + dec_res_s16

        dec_s16_cls = self.dec_s16_cls(dec_s16)

        target = self.get_target(dec_s16, target_key)
        targets.append(target)
        out_cls.append(dec_s16_cls)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        if self.training:
            keep_s16 += target

        # Remove voxels s16
        if keep_s16.sum() > 0 and self.pruning_choice[0]:
            dec_s16 = self.pruning(dec_s16, keep_s16)            

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_res_s8 = self.dec_res_block_s8_0(dec_s8)
        dec_s8 = dec_s8 + dec_res_s8
        dec_res_s8 = self.dec_res_block_s8_1(dec_s8)
        dec_s8 = dec_s8 + dec_res_s8

        dec_s8_cls = self.dec_s8_cls(dec_s8)

        target = self.get_target(dec_s8, target_key)
        targets.append(target)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        if self.training:
            keep_s8 += target

        # Remove voxels s16
        if keep_s8.sum() > 0 and self.pruning_choice[1]:
            dec_s8 = self.pruning(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_res_s4 = self.dec_res_block_s4_0(dec_s4)
        dec_s4 = dec_s4 + dec_res_s4
        dec_res_s4 = self.dec_res_block_s4_1(dec_s4)
        dec_s4 = dec_s4 + dec_res_s4

        dec_s4_cls = self.dec_s4_cls(dec_s4)

        target = self.get_target(dec_s4, target_key)
        targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        if self.training:
            keep_s4 += target

        # Remove voxels s4
        if keep_s4.sum() > 0 and self.pruning_choice[2]:
            dec_s4 = self.pruning(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_res_s2 = self.dec_res_block_s2_0(dec_s2)
        dec_s2 = dec_s2 + dec_res_s2
        dec_res_s2 = self.dec_res_block_s2_1(dec_s2)
        dec_s2 = dec_s2 + dec_res_s2

        dec_s2_cls = self.dec_s2_cls(dec_s2)

        target = self.get_target(dec_s2, target_key)
        targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        if self.training:
            keep_s2 += target

        # Remove voxels s2
        if keep_s2.sum() > 0 and self.pruning_choice[3]:
            dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_res_s1 = self.dec_res_block_s1_0(dec_s1)
        dec_s1 = dec_s1 + dec_res_s1
        dec_res_s1 = self.dec_res_block_s1_1(dec_s1)
        dec_s1 = dec_s1 + dec_res_s1

        dec_s1_cls = self.dec_s1_cls(dec_s1)

        target = self.get_target(dec_s1, target_key)
        targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        # Remove voxels s1
        if keep_s1.sum() > 0 and self.pruning_choice[4]:
            dec_s1 = self.pruning(dec_s1, keep_s1)

        shape_out = self.se_out(dec_s1)

        return out_cls, targets, shape_out


class D_Seg(nn.Module):
    def __init__(self, config={}):
        nn.Module.__init__(self)
        self.config = config

        if self.config['TRAIN']['D_TRAIN']['D_input'] == 'radial_height':
            in_channels = 2
        else:
            in_channels = 1

        self.classify = Seg_Net(config, in_channels=in_channels, out_channels=self.config['TRAIN']['class_count'])

    def forward(self, data_in):
        class_out0 = self.classify(data_in)

        return class_out0


class D_SSC(nn.Module):
    def __init__(self, config={}):
        nn.Module.__init__(self)
        self.config = config

        self.ssc = SSC_Net(config)

    def forward(self, class_out0):
        if self.training:
            batch_size = self.config['DATA_IO']['train_batch_size']
        else:
            batch_size = self.config['DATA_IO']['valid_batch_size']

        sparse_in = class_out0
        feature_dim = self.config['TRAIN']['class_count']

        dense_input = sparse_in.dense(shape=torch.Size([batch_size,feature_dim,256,256,32]))[0]

        class_out1 = self.ssc(dense_input)

        return class_out1


class D_shape(nn.Module):
    def __init__(self, in_channels, config={}):
        nn.Module.__init__(self)
        self.config = config

        self.shape = Com_Net(config, in_channels)

    def forward(self, class_out0, target_key):
        shape_in = class_out0

        out_cls, targets, shape_out = self.shape(shape_in, target_key)

        return out_cls, targets, shape_out


class G_siren(nn.Module):
    def __init__(self, config={}):
        nn.Module.__init__(self)
        self.config = config

        if config['TRAIN']['encode_xyz'] == True:
            xyz_dim = 3 * (config['TRAIN']['inc_input'] + config['TRAIN']['encode_levels'] * 2)
        else:
            xyz_dim = 3
        self.G_model = SingleBVPNet(out_features=1,
                                    type=config['TRAIN']['G_TRAIN']['nonlinearity'],
                                    in_features=(xyz_dim+config['TRAIN']['shape_embedding_size']), 
                                    hidden_features=config['TRAIN']['G_TRAIN']['hidden_features'],
                                    num_hidden_layers=config['TRAIN']['G_TRAIN']['num_hidden_layers'],
                                    config=config)
    def forward(self, shapes, coords):

        g_model_output = self.G_model(shapes, coords)

        return g_model_output


class G_label(nn.Module):
    def __init__(self, config={}):
        nn.Module.__init__(self)
        self.config = config

        if config['TRAIN']['encode_xyz'] == True:
            xyz_dim = 3 * (config['TRAIN']['inc_input'] + config['TRAIN']['encode_levels'] * 2)
        else:
            xyz_dim = 3
        self.G_model = SingleBVPNet(out_features=20,
                                    type=config['TRAIN']['G_TRAIN']['nonlinearity'],
                                    in_features=(xyz_dim+config['TRAIN']['shape_embedding_size']), 
                                    hidden_features=config['TRAIN']['G_TRAIN']['hidden_features'],
                                    num_hidden_layers=config['TRAIN']['G_TRAIN']['num_hidden_layers'],
                                    config=config)
    def forward(self, shapes, coords):

        g_model_output = self.G_model(shapes, coords)

        return g_model_output


