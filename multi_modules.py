import torch
from torch import nn
import torch.nn.functional as F
import math
from time import perf_counter
import time
import kilonerf_cuda
from utils import *
from torch.distributions.bernoulli import Bernoulli

# Only this function had to be changed to account for multi networks (weight tensors have aditionally a network dimension)
def _calculate_fan_in_and_fan_out(tensor):
    fan_in = tensor.size(-1)
    fan_out = tensor.size(-2)
    return fan_in, fan_out
    
# All of the above functions are copy pasted from PyTorch's codebase. This is nessecary because of the adapted fan in computation
def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
        
def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-a, a)

def xavier_normal_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0., std)
        
class MultiNetworkFourierEmbedding(nn.Module):
    def __init__(self, num_networks, num_input_channels, num_frequencies,):
        super(MultiNetworkFourierEmbedding, self).__init__()
        
        max_frequency = num_frequencies - 1
        self.frequency_bands = 2.**torch.linspace(0., max_frequency, steps=num_frequencies)
        self.num_frequencies = num_frequencies
        self.num_output_channels = (2 * num_frequencies + 1) * num_input_channels
        self.num_networks = num_networks
    
    def forward(self, x, implementation='pytorch', num_blocks=46, num_threads=512):
        # x: num_networks x batch_size x num_input_channels
        batch_size, num_input_channels = x.size(1), x.size(2)
        if implementation == 'pytorch':
            x = x.unsqueeze(3).expand(self.num_networks, batch_size, num_input_channels, 2 * self.num_frequencies + 1).contiguous()
            x[:,:,:, 1:1+self.num_frequencies] = x[:,:,:, 0].unsqueeze(3) * self.frequency_bands.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(x)
            x[:,:,:, 1+self.num_frequencies:] = x[:,:,:, 1:1+self.num_frequencies]
            x[:,:,:, 1:1+self.num_frequencies] = torch.cos(x[:,:,:, 1:1+self.num_frequencies])
            x[:,:,:, 1+self.num_frequencies:] = torch.sin(x[:,:,:, 1+self.num_frequencies:])
        else:
            self.frequency_bands = self.frequency_bands.to(x)
            x = kilonerf_cuda.compute_fourier_features(x.contiguous().view(-1), self.frequency_bands, num_blocks, num_threads, implementation)
        return x.view(self.num_networks, batch_size, -1)

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
   
# For hard parameter sharing    
class SharedLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, nonlinearity='leaky_relu', weight_initialization_method='kaiming_uniform', bias_initialization_method='standard'):
        super(SharedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.nonlinearity = nonlinearity
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initialization_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
        elif self.weight_initialization_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
        elif self.weight_initialization_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(self.nonlinearity))
        elif self.weight_initialization_method == 'xavier_normal':
            nn.init.xavier_normal_(self.weight, gain=nn.init.calculate_gain(self.nonlinearity))
        if self.bias is not None:
            if self.bias_initialization_method == 'standard':
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            elif self.bias_initialization_method == 'zeros':
                nn.init.zeros_(self.bias)

    # batch_size_per_network is a dummy argument
    def forward(self, input, batch_size_per_network=None):
        has_network_dim = len(list(input.size())) == 3
        if has_network_dim: # ignore network dimension
            num_networks = input.size(0)
            input = input.view(-1, self.in_features)
        out = F.linear(input, self.weight, self.bias)
        if has_network_dim:
            out = out.view(num_networks, -1, self.out_features)
        return out

def naive_multimatmul(biases, input_vectors, weights, out_features, in_features, batch_size_per_network):
    num_points = len(input_vectors)
    num_networks = len(biases)
    result_naive = torch.empty(num_points, out_features, device=torch.device('cuda'))
    start_index = 0
    for network_index in range(num_networks):
        end_index = start_index + batch_size_per_network[network_index].item()
        #torch.matmul(input_vectors[start_index:end_index], weights[network_index], out=result_naive[start_index:end_index])
        torch.addmm(biases[network_index], input_vectors[start_index:end_index], weights[network_index], out=result_naive[start_index:end_index])
        start_index = end_index
    return result_naive

def naive_multimatmul_differentiable(biases, input_vectors, weights, out_features, in_features, batch_size_per_network):
    num_points = len(input_vectors)
    num_networks = len(biases)
    result_naive = torch.empty(num_points, out_features, device=torch.device('cuda'))
    start_index = 0
    for network_index in range(num_networks):
        end_index = start_index + batch_size_per_network[network_index].item()
        temp_res = torch.addmm(biases[network_index], input_vectors[start_index:end_index], weights[network_index])
        result_naive[start_index:end_index] = temp_res
        start_index = end_index
    return result_naive
    
class AddMultiMatMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, biases, input_vectors, weights, out_features, in_features, batch_size_per_network, group_limits, aux_index, aux_index_backward):
        ctx.save_for_backward(biases, input_vectors, weights, batch_size_per_network)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.group_limits = group_limits
        ctx.aux_index = aux_index
        ctx.aux_index_backward = aux_index_backward
        return kilonerf_cuda.multimatmul_magma_grouped_static(biases, input_vectors, weights,
            out_features, in_features, batch_size_per_network, 4, 1024, group_limits, aux_index)

    @staticmethod
    def backward(ctx, grad_output):
        biases, input_vectors, weights, batch_size_per_network = ctx.saved_tensors
        
        #print(biases)
        #print(input_vectors)
        #print(weights)
        #print(batch_size_per_network)
        
        grad_output = grad_output.contiguous()
        
        grad_biases = None
        grad_input_vectors = None
        grad_weights = None
        
        grad_biases = kilonerf_cuda.multi_row_sum_reduction(grad_output, batch_size_per_network)
        
        grad_input_vectors = kilonerf_cuda.multimatmul_magma_grouped_static_without_bias_transposed_weights(biases, grad_output, weights,
            ctx.in_features, ctx.out_features, batch_size_per_network, 4, 1024, ctx.group_limits, ctx.aux_index_backward)
            
        grad_weights = kilonerf_cuda.multimatmul_A_transposed(input_vectors, grad_output, batch_size_per_network)
        
        return grad_biases, grad_input_vectors, grad_weights, None, None, None, None, None, None

class MultiNetworkLinear(nn.Module):
    rng_state = None

    def __init__(self, num_networks, in_features, out_features, nonlinearity='leaky_relu',
        bias=True, implementation='bmm', nonlinearity_params=None, use_same_initialization_for_all_networks=False,
        network_rng_seed=None, weight_initialization_method='kaiming_uniform', bias_initialization_method='standard'):
        
        super(MultiNetworkLinear, self).__init__()
        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features
        self.implementation = implementation
        self.use_same_initialization_for_all_networks = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        # weight is created in reset_parameters()
        if self.implementation.startswith('multimatmul'):
            self.group_limits = [2048, 1024] # tunable
            self.aux_index = kilonerf_cuda.init_multimatmul_magma_grouped(self.num_networks, self.out_features, self.in_features, self.group_limits)
            if self.implementation == 'multimatmul_differentiable':
                # out_features and in_features are interchanged
                self.aux_index_backward = kilonerf_cuda.init_multimatmul_magma_grouped(self.num_networks, self.in_features, self.out_features, self.group_limits)
        self.nonlinearity = nonlinearity
        self.nonlinearity_params = nonlinearity_params
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_networks, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight = nn.Parameter(torch.Tensor(self.num_networks, self.out_features, self.in_features))
        
        # Use a seperate RNG seed for network initalization to be able to keep
        # other random aspects (i.e. batch sampling) fixed, while varying network initalization
        if self.network_rng_seed is not None:
            previous_rng_state = torch.random.get_rng_state()
            if MultiNetworkLinear.rng_state is None:
                torch.random.manual_seed(self.network_rng_seed)
            else:
                torch.random.set_rng_state(MultiNetworkLinear.rng_state)
                
        if self.nonlinearity != 'sine':
            if self.weight_initialization_method == 'kaiming_uniform':
                kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
            elif self.weight_initialization_method == 'kaiming_normal':
                kaiming_normal_(self.weight, a=math.sqrt(5), nonlinearity=self.nonlinearity)
            elif self.weight_initialization_method == 'xavier_uniform':
                xavier_uniform_(self.weight, gain=calculate_gain(self.nonlinearity))
            elif self.weight_initialization_method == 'xavier_normal':
                xavier_normal_(self.weight, gain=calculate_gain(self.nonlinearity))
            if self.bias is not None:
                if self.bias_initialization_method == 'standard':
                    fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(self.bias, -bound, bound)
                elif self.bias_initialization_method == 'zeros':
                    nn.init.zeros_(self.bias)
        else: # For SIREN
            c, w0, is_first = self.nonlinearity_params['c'], self.nonlinearity_params['w0'], self.nonlinearity_params['is_first']
            w_std = (1 / self.in_features) if is_first else (math.sqrt(c / self.in_features) / w0)
            nn.init.uniform_(self.weight, -w_std, w_std)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -w_std, w_std)
                
        if self.network_rng_seed is not None:
            MultiNetworkLinear.rng_state = torch.random.get_rng_state()
            torch.random.set_rng_state(previous_rng_state)
        
        if self.use_same_initialization_for_all_networks:
            with torch.no_grad():
                self.weight[1:] = self.weight[0]
                self.bias[1:] = self.bias[0] 
        
        if 'multimatmul' in self.implementation:
            self.weight.data = self.weight.data.view(self.num_networks, self.in_features, self.out_features).contiguous()
            
    def forward(self, x, batch_size_per_network=None, bias=None, weight=None):
        # For testing purposes override weight and bias
        if bias is not None:
            self.bias = bias
        if weight is not None:
            self.weight = weight
        if self.implementation == 'multimatmul':
            # x = num_points x in_features
            return kilonerf_cuda.multimatmul_magma_grouped_static(self.bias, x.contiguous(), self.weight,
                self.out_features, self.in_features, batch_size_per_network, 4, 1024, self.group_limits, self.aux_index)
        elif self.implementation == 'multimatmul_differentiable':
            return AddMultiMatMul.apply(self.bias, x.contiguous(), self.weight, self.out_features, self.in_features, batch_size_per_network, self.group_limits,
                self.aux_index, self.aux_index_backward)
        elif self.implementation == 'naive_multimatmul_differentiable':
            return naive_multimatmul_differentiable(self.bias, x, self.weight, self.out_features, self.in_features, batch_size_per_network)
        else:
            # x = num_networks x batch_size x in_features
            batch_size = x.size(1)
            if self.num_networks > 1:
                if self.implementation == 'bmm':
                    weight_transposed = self.weight.permute(0, 2, 1) # num_networks x in_features x out_features
                    
                    # num_networks x batch_size x in_features @ num_networks x in_features x out_features = num_networks x batch_size x out_features
                    product = torch.bmm(x, weight_transposed)
                    bias_view = self.bias.unsqueeze(1)
                elif self.implementation == 'matmul':
                    input_view = x.unsqueeze(3) # num_networks x batch_size x in_features x 1
                    weight_view = self.weight.unsqueeze(1) # num_networks x 1 x out_features x in_features
                    product = torch.matmul(weight_view, input_view).squeeze(3) # num_networks x batch_size x out_features
                    bias_view = self.bias.unsqueeze(1) # num_networks x 1 x out_features
                result = product + bias_view # (num_networks * batch_size) x out_features
            else:
                input_view = x.squeeze(0)
                weight_view = self.weight.squeeze(0)
                bias_view = self.bias.squeeze(0)
                result = F.linear(input_view, weight_view, bias_view)
            return result.view(self.num_networks, batch_size, self.out_features)
            

def extract_linears(network):
    linears, shared_linears = [], []
    for module in network.modules():
        if isinstance(module, MultiNetworkLinear):
            linears.append(module)
        if isinstance(module, SharedLinear):
            shared_linears.append(module)
    return linears, shared_linears

class MultiNetwork(nn.Module):
    def __init__(self, num_networks, num_position_channels, num_direction_channels, num_output_channels, hidden_layer_size, num_hidden_layers, refeed_position_index=None, late_feed_direction=False,
        direction_layer_size=None, nonlinearity='relu', nonlinearity_initalization='pass_leaky_relu', use_single_net=False, linear_implementation='bmm', use_same_initialization_for_all_networks=False,
        network_rng_seed=None, weight_initialization_method='kaiming_uniform', bias_initialization_method='standard', alpha_rgb_initalization='updated_yenchenlin', use_hard_parameter_sharing_for_color=False,
        view_dependent_dropout_probability=-1, use_view_independent_color=False):
        super(MultiNetwork, self).__init__()
        
        self.num_networks = num_networks
        self.num_position_channels = num_position_channels
        self.num_direction_channels = num_direction_channels
        self.num_output_channels = num_output_channels
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.refeed_position_index = refeed_position_index
        self.late_feed_direction = late_feed_direction
        self.direction_layer_size = direction_layer_size
        self.nonlinearity = nonlinearity
        self.nonlinearity_initalization = nonlinearity_initalization # 'pass_leaky_relu', 'pass_actual_nonlinearity'
        self.use_single_net = use_single_net
        self.linear_implementation = linear_implementation
        self.use_same_initialization_for_all_networks = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        self.alpha_rgb_initalization = alpha_rgb_initalization # 'updated_yenchenlin', 'pass_actual_nonlinearity'
        self.use_hard_parameter_sharing_for_color = use_hard_parameter_sharing_for_color
        self.view_dependent_dropout_probability = view_dependent_dropout_probability
        self.use_view_independent_color = use_view_independent_color
        
        nonlinearity_params = {}
        if nonlinearity == 'sigmoid':
            self.activation = nn.Sigmoid()
        if nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        if nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        if nonlinearity == 'sine':
            nonlinearity_params = {'w0': 30., 'c': 6., 'is_first': True}
            self.activation = Sine(nonlinearity_params['w0'])
            
        # TODO: weight_initalization_method and bias_initalization_method are beeing ignored
        def linear_layer(in_features, out_features, actual_nonlinearity, use_hard_parameter_sharing=False):
            if self.nonlinearity_initalization == 'pass_actual_nonlinearity': # proper way of doing things
                passed_nonlinearity = actual_nonlinearity 
            elif self.nonlinearity_initalization == 'pass_leaky_relu': # to reproduce the old behaviour (doesn't make a lot of sense though)
                passed_nonlinearity = 'leaky_relu'
            if not use_hard_parameter_sharing:
                return MultiNetworkLinear(self.num_networks, in_features, out_features,
                    nonlinearity=passed_nonlinearity, nonlinearity_params=nonlinearity_params, implementation=linear_implementation,
                    use_same_initialization_for_all_networks=use_same_initialization_for_all_networks, network_rng_seed=network_rng_seed)
            else:
                print('Using hard parameter sharing')
                return SharedLinear(in_features, out_features, bias=True, nonlinearity=passed_nonlinearity)

        if self.late_feed_direction:
            self.pts_linears = [linear_layer(self.num_position_channels, self.hidden_layer_size, self.nonlinearity)]
            nonlinearity_params = nonlinearity_params.copy().update({'is_first': False})
            for i in range(self.num_hidden_layers - 1):
                if i == self.refeed_position_index:
                    new_layer = linear_layer(self.hidden_layer_size + self.num_position_channels, self.hidden_layer_size, self.nonlinearity)
                else:
                    new_layer = linear_layer(self.hidden_layer_size, self.hidden_layer_size, self.nonlinearity)
                self.pts_linears.append(new_layer)
            self.pts_linears = nn.ModuleList(self.pts_linears)
            self.direction_layer = linear_layer(self.num_direction_channels + self.hidden_layer_size, self.direction_layer_size, self.nonlinearity, self.use_hard_parameter_sharing_for_color)
            
            if self.use_view_independent_color:
                feature_output_size = self.hidden_layer_size + 4 # + RGBA
            else:
                feature_output_size = self.hidden_layer_size
            self.feature_linear = linear_layer(self.hidden_layer_size, feature_output_size, 'linear')
            # In the updated yenchenlin implementation which follows now closely the original tensorflow implementation
            # 'linear' is passed to these two layers, but it also makes sense to pass the actual nonlinearites here
            if not self.use_view_independent_color:
                self.alpha_linear = linear_layer(self.hidden_layer_size, 1, 'linear' if self.alpha_rgb_initalization == 'updated_yenchenlin' else 'relu')
            self.rgb_linear = linear_layer(self.direction_layer_size, 3, 'linear' if self.alpha_rgb_initalization == 'updated_yenchenlin' else 'sigmoid',
                self.use_hard_parameter_sharing_for_color)
                
            self.view_dependent_parameters = list(self.direction_layer.parameters()) + list(self.rgb_linear.parameters()) # needed for L2 regularization only on the view-dependent part of the network
            
            if self.view_dependent_dropout_probability > 0:
                self.dropout_after_feature = nn.Dropout(self.view_dependent_dropout_probability)
                self.dropout_after_direction_layer = nn.Dropout(self.view_dependent_dropout_probability)
            
        else:
            layers = [linear_layer(self.num_position_channels + self.num_direction_channels, self.hidden_layer_size), self.activation]
            nonlinearity_params = nonlinearity_params.copy().update({'is_first': False})
            for _ in range(self.num_hidden_layers): # TODO: should be also self.num_hidden_layers - 1
                layers += [linear_layer(self.hidden_layer_size, self.hidden_layer_size), self.activation]
            layers += [linear_layer(self.hidden_layer_size, self.num_output_channels)]
            self.layers = nn.Sequential(*layers)
    
    # needed for fused kernel
    def serialize_params(self):
        # fused kernel expects IxO matrix instead of OxI matrix
        def process_weight(w):
            return w.reshape(self.num_networks, -1)
    
        self.serialized_params = []
        for l in self.pts_linears:
            self.serialized_params += [l.bias, process_weight(l.weight)]
            
        self.serialized_params.append(torch.cat([self.alpha_linear.bias, self.feature_linear.bias], dim=1))
        self.serialized_params.append(process_weight(torch.cat([self.alpha_linear.weight, self.feature_linear.weight], dim=2)))
        for l in [self.direction_layer, self.rgb_linear]:
            self.serialized_params += [l.bias, process_weight(l.weight)]
        self.serialized_params = torch.cat(self.serialized_params, dim=1).contiguous()

    # random_directions will be used for regularizing the view-independent color
    def forward(self, x, batch_size_per_network=None, random_directions=None):
        if self.late_feed_direction:
            if isinstance(x, list):
                positions, directions = x
                # frees memory of inputs
                x[0] = None 
                x[1] = None
            else:
                positions, directions = torch.split(x, [self.num_position_channels, self.num_direction_channels], dim=-1)
            h = positions
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h, batch_size_per_network)
                PerfMonitor.add('pts_linears ' + str(i), ['network query', 'matmul'])
                h = self.activation(h)
                PerfMonitor.add('activation ' + str(i), ['network query', 'matmul'])
                if i == self.refeed_position_index:
                    h = torch.cat([positions, h], -1)
                    PerfMonitor.add('cat[positions, h]', ['network query', ])
            del positions
            if not self.use_view_independent_color:
                alpha = self.alpha_linear(h, batch_size_per_network)
                PerfMonitor.add('alpha_linear', ['network query', 'matmul'])
            feature = self.feature_linear(h, batch_size_per_network) # TODO: investigate why they don't use an activation function on top of feature layer!
            if self.view_dependent_dropout_probability > 0:
                feature =  self.dropout_after_feature(feature)
            if self.use_view_independent_color:
               rgb_view_independent, alpha, feature = torch.split(feature, [3, 1, self.hidden_layer_size], dim=-1)
            PerfMonitor.add('feature_linear', ['network query', 'matmul'])
            del h
            
            # Regularizing the view-independent color to be the mean of view-dependent colors sampled at some random directions
            if random_directions is not None:
                assert self.use_view_independent_color == True, 'this regularization only makes sense if we output a view-independent color'
                num_random_directions = random_directions.size(0)
                batch_size = feature.size(0)
                feature_size = feature.size(1)
                feature = feature.repeat(1, num_random_directions + 1).view(-1, feature_size)
                random_directions = random_directions.repeat(batch_size, 1).view(batch_size, num_random_directions, -1)
                directions = torch.cat([directions.unsqueeze(1), random_directions], dim=1).view(batch_size * (num_random_directions + 1), -1)
                batch_size_per_network = (num_random_directions + 1) * batch_size_per_network

            
            # View-dependent part of the network:
            h = torch.cat([feature, directions], -1)
            PerfMonitor.add('cat[feature, directions]', ['network query'])
            del feature
            del directions
            h = self.direction_layer(h, batch_size_per_network)
            PerfMonitor.add('direction_linear', ['network query', 'matmul'])
            h = self.activation(h)
            if self.view_dependent_dropout_probability > 0:
                h = self.dropout_after_direction_layer(h)
            PerfMonitor.add('direction activation', ['network query'])
            rgb = self.rgb_linear(h, batch_size_per_network)
            PerfMonitor.add('rgb_linear', ['network query', 'matmul'])
            del h

            if self.use_view_independent_color:
                if random_directions is None:
                    rgb = rgb + rgb_view_independent
                else:
                    mean_rgb = rgb.view(batch_size, num_random_directions + 1, 3)
                    mean_rgb = mean_rgb + rgb_view_independent.unsqueeze(1)
                    rgb = mean_rgb[:, 0]
                    mean_rgb = mean_rgb.mean(dim=1)
                    mean_regularization_term = torch.abs(mean_rgb - rgb_view_independent).mean()
                    del mean_rgb
                del rgb_view_independent
                PerfMonitor.add('rgb + rgb_view_independent', ['network query'])
                
            result = torch.cat([rgb, alpha], -1)
            PerfMonitor.add('cat[rgb, alpha]', ['network query'])
            
            if random_directions is not None:
                return result, mean_regularization_term
            else:
                return result
        else:
            return self.layers(x)
            

    def extract_single_network(self, network_index):
        single_network = MultiNetwork(1, self.num_position_channels, self.num_direction_channels, self.num_output_channels,
            self.hidden_layer_size, self.num_hidden_layers, self.refeed_position_index, self.late_feed_direction,
            self.direction_layer_size, self.nonlinearity, self.nonlinearity_initalization, self.use_single_net,
            use_hard_parameter_sharing_for_color=self.use_hard_parameter_sharing_for_color,
            view_dependent_dropout_probability=self.view_dependent_dropout_probability,
            use_view_independent_color=self.use_view_independent_color)
      
        multi_linears, multi_shared_linears = extract_linears(self)
        single_linears, single_shared_linears = extract_linears(single_network)
        with torch.no_grad():
            for single_linear, multi_linear in zip(single_linears, multi_linears):
                single_linear.weight.data[0] = multi_linear.weight.data[network_index]
                single_linear.bias.data[0] = multi_linear.bias.data[network_index]
                   
            for single_shared_linear, multi_shared_linear in zip(single_shared_linears, multi_shared_linears):
                single_shared_linear.weight.data = multi_shared_linear.weight.data
                single_shared_linear.bias.data = multi_shared_linear.bias.data
            
        return single_network
    
    # Just for the unit test
    def _extract_single_network(self, network_index):
        def copy_parameters(network_index, linear, multi_network_linear):
            with torch.no_grad():
                linear.weight.data[:] = multi_network_linear.weight.data[network_index]
                linear.bias.data[:] = multi_network_linear.bias.data[network_index]
                
        input_layer = nn.Linear(self.num_input_channels, self.hidden_layer_size)
        layer_index = 0
        copy_parameters(network_index, input_layer, self.layers[layer_index])
        single_network_layers = [input_layer, nn.ReLU()]
        layer_index = 2
        for _ in range(self.num_hidden_layers):
            hidden_layer = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
            copy_parameters(network_index, hidden_layer, self.layers[layer_index])
            single_network_layers += [hidden_layer, nn.ReLU()]
            layer_index += 2
        output_layer = nn.Linear(self.hidden_layer_size, self.num_output_channels)
        copy_parameters(network_index, output_layer, self.layers[layer_index])
        single_network_layers += [output_layer]
        return  nn.Sequential(*single_network_layers)

# semi-fast querying, differentiable, supports abitrary ray batches as input
def query_multi_network(multi_network, domain_mins, domain_maxs, points, directions,
    position_fourier_embedding, direction_fourier_embedding, occupancy_grid, debug_network_color_map, use_network_jittering, random_directions, cfg):
    num_rays = points.size(0)
    num_samples = points.size(1)
    num_networks = multi_network.num_networks
    
    points_flat = points.view(-1, 3)
    res = cfg['fixed_resolution']
    fixed_resolution = torch.tensor(res, dtype=torch.long, device=points_flat.device)
    network_strides = torch.tensor([res[2] * res[1], res[2], 1], dtype=torch.long, device=points_flat.device) # assumes row major ordering
    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(points_flat.device)
    global_domain_size = global_domain_max - global_domain_min
    voxel_size = global_domain_size / fixed_resolution
    point_indices_3d = ((points_flat - global_domain_min) / voxel_size).to(network_strides)
    point_indices = (point_indices_3d * network_strides).sum(dim=1)
    
    # just for debugging
    if 'block_size' in cfg:
        assigned_networks_per_block = point_indices.view(-1, cfg['block_size'][0], cfg['block_size'][1], num_samples)
        
        histogram = [0 for _ in range(1024)]
        for i in range(assigned_networks_per_block.shape[0]):
            for j in range(0, assigned_networks_per_block.shape[-1], cfg['block_size'][2]):
                unique_assigned_networks, counts = torch.unique(assigned_networks_per_block[i, :, :, j:j+cfg['block_size'][2]], return_counts=True)
                print(counts.cpu().tolist())
                histogram[len(unique_assigned_networks)] += 1
                print(histogram[1:10])
        exit()
    
    if not use_network_jittering:
        del point_indices_3d

    # define a mapping to filter empty regions: 0 -> -1, 1 -> 1, 2 -> 2, 3 -> -1, 4 -> -1
    if occupancy_grid is not None:
        res = cfg['occupancy']['resolution']
        occupancy_resolution = torch.tensor(res, dtype=torch.long, device=points_flat.device)
        strides = torch.tensor([res[2] * res[1], res[2], 1], dtype=torch.long, device=points_flat.device) # assumes row major ordering
        voxel_size = global_domain_size / occupancy_resolution
        occupancy_indices = ((points_flat - global_domain_min) / voxel_size).to(torch.long)
        torch.max(torch.tensor([0, 0, 0], device=points_flat.device), occupancy_indices, out=occupancy_indices)
        torch.min(occupancy_resolution - 1, occupancy_indices, out=occupancy_indices)
        occupancy_indices = (occupancy_indices * strides).sum(dim=1)
        
        point_in_occupied_space = occupancy_grid[occupancy_indices]
        del occupancy_indices
    
    # Filtering points outside global domain
    epsilon = 0.001
    active_samples_mask = torch.logical_and((points_flat > global_domain_min + epsilon).all(dim=1), (points_flat < global_domain_max - epsilon).all(dim=1))
    if occupancy_grid is not None:
        active_samples_mask = torch.logical_and(active_samples_mask, point_in_occupied_space)
        del point_in_occupied_space
    proper_index = torch.logical_and(point_indices >= 0, point_indices < num_networks) # probably this is not needed if we check for points_flat <= global_domain_max
    active_samples_mask = torch.nonzero(torch.logical_and(active_samples_mask, proper_index), as_tuple=False).squeeze()
    del proper_index
    
    filtered_point_indices = point_indices[active_samples_mask]
    del point_indices
    
    # Unused regularization technique
    if use_network_jittering:
        filtered_point_indices_3d = point_indices_3d[active_samples_mask]
        del point_indices_3d
    
        # global to local conversion to calculate distance to neighbor
        domain_mins_reordered = domain_mins[filtered_point_indices]
        domain_maxs_reordered = domain_maxs[filtered_point_indices]
        local_points = points_flat[active_samples_mask]
        local_points = 2 * (local_points - domain_mins_reordered) / (domain_maxs_reordered - domain_mins_reordered) - 1 # coords in [-1, 1]
        del domain_mins_reordered, domain_maxs_reordered
        
        # The closer we are to the border the higher the probability of changing the assigned network to the respective bordering network
        pos_dist = 1 - local_points # distance to "positive" borders
        neg_dist = local_points + 1 # distance to "negative" borders
        del local_points
        pos_dist = cfg['network_jitter']['prob_at_border'] * torch.exp(-cfg['network_jitter']['dropoff_rate'] * pos_dist)
        neg_dist = cfg['network_jitter']['prob_at_border'] * torch.exp(-cfg['network_jitter']['dropoff_rate'] * neg_dist)
        pos_offsets = Bernoulli(pos_dist).sample().long()
        neg_offsets = Bernoulli(neg_dist).sample().long()
        del pos_dist, neg_dist

        # Make sure that we are not jittering out of bounds
        pos_offsets[filtered_point_indices_3d[:, 0] == fixed_resolution[0] - 1, 0] = 0
        pos_offsets[filtered_point_indices_3d[:, 1] == fixed_resolution[1] - 1, 1] = 0
        pos_offsets[filtered_point_indices_3d[:, 2] == fixed_resolution[2] - 1, 2] = 0
        neg_offsets[filtered_point_indices_3d[:, 0] == 0, 0] = 0
        neg_offsets[filtered_point_indices_3d[:, 1] == 0, 1] = 0
        neg_offsets[filtered_point_indices_3d[:, 2] == 0, 2] = 0

        # Jittering
        filtered_point_indices_3d += pos_offsets
        filtered_point_indices_3d -= neg_offsets
        del pos_offsets, neg_offsets
        filtered_point_indices = (filtered_point_indices_3d * network_strides).sum(dim=1) # convert to flat indices again
        del filtered_point_indices_3d

    # Sort according to network
    filtered_point_indices, reorder_indices = torch.sort(filtered_point_indices)
    
    # make sure that also batch sizes are given for networks which are queried 0 points
    contained_nets, batch_size_per_network_incomplete = torch.unique_consecutive(filtered_point_indices, return_counts=True)
    del filtered_point_indices
    batch_size_per_network = torch.zeros(num_networks, device=points_flat.device, dtype=torch.long)
    batch_size_per_network[contained_nets] = batch_size_per_network_incomplete
    batch_size_per_network = batch_size_per_network.cpu()
    
    
    # Reordering
    directions_flat = directions.unsqueeze(1).expand(points.size()).reshape(-1, 3)
    points_reordered = points_flat[active_samples_mask] 
    directions_reordered = directions_flat[active_samples_mask]
    del points_flat, directions_flat
    # reorder so that points handled by the same network are packed together in the list of points
    points_reordered = points_reordered[reorder_indices]
    directions_reordered = directions_reordered[reorder_indices]
    PerfMonitor.add('reorder', ['reorder and backorder'])
    
    num_points_to_process = points_reordered.size(0) if points_reordered.ndim > 0 else 0
    print("#points to process:", num_points_to_process, flush=True)
    if num_points_to_process == 0:
        return torch.zeros(num_rays, num_samples, 4, dtype=torch.float, device=points_reordered.device)
            
    # Convert global to local coordinates
    if not has_flag(cfg, 'use_global_coordinates'):
        kilonerf_cuda.global_to_local(points_reordered, domain_mins, domain_maxs, batch_size_per_network, 1, 64)
        PerfMonitor.add('global to local', ['input transformation'])
    
    # Fourier features
    fourier_embedding_implementation = 'custom_kernel_v2' # pytorch
    if position_fourier_embedding is not None:
        embedded_points = position_fourier_embedding(points_reordered.unsqueeze(0), implementation=fourier_embedding_implementation).squeeze(0)
    else:
        embedded_points = points_reordered
    del points_reordered
    if direction_fourier_embedding is not None:
        embedded_dirs = direction_fourier_embedding(directions_reordered.unsqueeze(0), implementation=fourier_embedding_implementation).squeeze(0)
    else:
        embedded_dirs = directions_reordered
    del directions_reordered
    embedded_points_and_dirs = [embedded_points, embedded_dirs]
    del embedded_points
    del embedded_dirs
    PerfMonitor.add('fourier features', ['input transformation'])

    # Network query
    raw_outputs = multi_network(embedded_points_and_dirs, batch_size_per_network, random_directions)
    if random_directions is not None:
        raw_outputs, mean_regularization_term = raw_outputs

    # For debugging we can visualize which networks are responsible for which regions
    # This was also used to render the teaser figure.
    if has_flag(cfg, 'render_debug_network_color_map'):
        end_idx = 0
        batch_size_per_network_list = batch_size_per_network.tolist()
        for network_index in range(multi_network.num_networks):
            res = cfg['fixed_resolution']
            ind = [(network_index // (res[2] * res[1])), (network_index // res[2]) % res[1], network_index % res[2]]
            start_idx = end_idx
            end_idx += batch_size_per_network_list[network_index]
            use_color_map = True
            if 'network_color_map_min' in cfg:
                for a, b in zip(ind, cfg['network_color_map_min']):
                    use_color_map = use_color_map and ind >= cfg['network_color_map_min']
            if 'network_color_map_max' in cfg:
                for a, b in zip(ind, cfg['network_color_map_max']):
                    use_color_map = use_color_map and ind <= cfg['network_color_map_max']
            if start_idx != end_idx and use_color_map:
                # assign random color to each network
                raw_outputs[start_idx:end_idx, :3] = debug_network_color_map[network_index]
    
    # Naive reordering is extremly fast even without any explicit measures to gurantee coherence => DeRF authors were telling lies
    raw_outputs_backordered = torch.empty_like(raw_outputs)
    raw_outputs_backordered[reorder_indices] = raw_outputs
    #raw_outputs_backordered = kilonerf_cuda.scatter_int32_float4(reorder_indices, raw_outputs)
    del raw_outputs
    raw_outputs_full = torch.zeros(num_rays * num_samples, 4, dtype=torch.float, device=raw_outputs_backordered.device)
    raw_outputs_full[active_samples_mask] = raw_outputs_backordered
    PerfMonitor.add('backorder', ['reorder and backorder'])
    
    raw_outputs_full = raw_outputs_full.view(num_rays, num_samples, -1)
    if random_directions is None:
        return raw_outputs_full
    else:
        return raw_outputs_full, mean_regularization_term
        
def build_multi_network_from_single_networks(single_networks, transpose_weight = True, linear_implementation = 'multimatmul', view_dependent_dropout_probability=-1):
    num_networks = len(single_networks)
    p = single_networks[0]
    
    try:
        use_hard_parameter_sharing_for_color = p.use_hard_parameter_sharing_for_color
    except AttributeError:
        use_hard_parameter_sharing_for_color = False
    
    try:
        use_view_independent_color = p.use_view_independent_color
    except AttributeError:
        use_view_independent_color = False
    
    # The initalization parameters do not need to be passed, because weights are overwritten anyhow
    multi_network = MultiNetwork(num_networks, p.num_position_channels, p.num_direction_channels, p.num_output_channels, p.hidden_layer_size, p.num_hidden_layers, p.refeed_position_index, p.late_feed_direction,
        p.direction_layer_size, p.nonlinearity, linear_implementation=linear_implementation, use_hard_parameter_sharing_for_color=use_hard_parameter_sharing_for_color,
        view_dependent_dropout_probability=view_dependent_dropout_probability, use_view_independent_color=use_view_independent_color)
 
    multi_linears, multi_shared_linears = extract_linears(multi_network)
    linears_per_network = [extract_linears(network) for network in single_networks]
    num_linear_layers = len(multi_linears)
    num_linear_layers_shared = len(multi_shared_linears)
    with torch.no_grad():
        for layer_index in range(num_linear_layers):
            for network_index in range(multi_network.num_networks):
                new_weight = linears_per_network[network_index][0][layer_index].weight.data[0]
                new_bias = linears_per_network[network_index][0][layer_index].bias.data[0]
                # new multimatmul implementation requires transposed weights: in_features x out_features
                if transpose_weight:
                    new_weight = new_weight.t()
                    #new_bias = new_bias.t()
                multi_linears[layer_index].weight.data[network_index] = new_weight
                multi_linears[layer_index].bias.data[network_index] = new_bias
         
        for layer_index in range(num_linear_layers_shared):
            new_weight = linears_per_network[0][1][layer_index].weight.data
            new_bias = linears_per_network[0][1][layer_index].bias.data
            multi_shared_linears[layer_index].weight.data = new_weight
            multi_shared_linears[layer_index].bias.data = new_bias
                
    return multi_network


def orig_nerf_vs_our_nerf():
    import architectures
    dev = torch.device('cuda')
    
    # NeRF hyperparameters
    multires = 10
    multires_views = 4
    i_embed = 0
    use_viewdirs = True
    N_importance = 0
    netdepth = 8
    netwidth = 256
    
    embed_fn, input_ch = architectures.get_embedder(multires, i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if use_viewdirs:
        embeddirs_fn, input_ch_views = architectures.get_embedder(multires_views, i_embed)
    output_ch = 5 if N_importance > 0 else 4
    skips = [4]
    
    torch.manual_seed(42)
    orig_nerf = architectures.NeRF(D=netdepth, W=netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=use_viewdirs).to(dev)
    
    torch.manual_seed(42)
    our_nerf = MultiNetwork(1, input_ch, input_ch_views, skips[0], netwidth, netdepth, 4, True, netwidth // 2).to(dev)
    
    batch_size = 8
    positions = torch.rand(batch_size, 3).to(dev)
    directions = torch.rand(batch_size, 3).to(dev)
    
    embedded_positions = embed_fn(positions)
    embedded_directions = embeddirs_fn(directions)
    points_and_dirs = torch.cat([embedded_positions, embedded_directions], -1)
    orig_out = orig_nerf(points_and_dirs)
    our_out = our_nerf(points_and_dirs.unsqueeze(0))
    print('outputs match:', torch.allclose(orig_out, our_out.squeeze()))
        
def test():
    # Test Fourier Embedding
    embedding = MultiNetworkFourierEmbedding(2, 2, 3)
    data = torch.tensor([[[2, 0.5], [3, 7]], [[10, 20], [30, 40]]], dtype=float)
    print(embedding(data), embedding(data).size())
    quit()

    dev = torch.device('cuda')
    num_networks = 16
    num_input_channels = 3
    num_output_channels = 2
    hidden_layer_size = 20
    num_hidden_layers  = 2

    multi_network = MultiNetwork(num_networks, num_input_channels, num_output_channels, hidden_layer_size, num_hidden_layers).to(dev)
    
    # Check if outputs match between multi network and single networks with same weights
    batch_size = 1000
    data = torch.rand(num_networks, batch_size, num_input_channels).to(dev)
    parameters_matching = True
    with torch.no_grad():
        multi_output = multi_network(data)
        single_output = torch.empty_like(multi_output)
        for network_index in range(num_networks):
            single_network = multi_network.extract_single_network(network_index).to(dev)
            
            for multi_layer, single_layer in zip(multi_network.layers, single_network):
                if isinstance(single_layer, nn.Linear):
                    parameters_matching = parameters_matching and torch.allclose(single_layer.weight.data, multi_layer.weight.data[network_index])\
                        and torch.allclose(single_layer.bias.data, multi_layer.bias.data[network_index])
            
            single_output[network_index] = single_network(data[network_index])
            
    print('parameters close before training:', parameters_matching)
    print('outputs close:', torch.allclose(multi_output, single_output, rtol=1e-02))
        
    # Check if weights match after one step of gradient descent
    targets = torch.rand(num_networks, batch_size, num_output_channels).to(dev)
    
    # One GD step with each single network
    start_time = perf_counter()
    single_networks = []
    for network_index in range(num_networks):
        loss_func = nn.MSELoss()
        single_network = multi_network.extract_single_network(network_index).to(dev)
        optimizer = torch.optim.Adam(single_network.parameters())
        optimizer.zero_grad()
        single_output = single_network(data[network_index])
        mean_squared_error = loss_func(single_output, targets[network_index])
        mean_squared_error.backward()
        optimizer.step()
        single_networks.append(single_network)
    single_network_duration = perf_counter() - start_time
    
    # One GD step with multi network
    start_time = perf_counter()
    loss_func = nn.MSELoss(reduction='none') # we need to reduce it ourselves: average over output channel (2) and batch dimension (1) but sum over network dimension (0)
    optimizer = torch.optim.Adam(multi_network.parameters())
    optimizer.zero_grad()
    multi_output = multi_network(data)
    squared_error = loss_func(multi_output, targets) # num_networks x batch_size x num_output_channels
    mean_squared_error = squared_error.mean(dim=2).mean(dim=1) # num_networks
    sum_of_mean_squared_error = mean_squared_error.sum()
    sum_of_mean_squared_error.backward()
    optimizer.step()
    multi_network_duration = perf_counter() - start_time
    
    # Check if weights are matching
    parameters_matching = True    
    for network_index, single_network in enumerate(single_networks):
        for multi_layer, single_layer in zip(multi_network.layers, single_network):
            if isinstance(single_layer, nn.Linear):
                parameters_matching = parameters_matching and torch.allclose(single_layer.weight.data, multi_layer.weight.data[network_index])\
                    and torch.allclose(single_layer.bias.data, multi_layer.bias.data[network_index])
    print('parameters close after training', parameters_matching)
    print('single network time: {}, multi network time: {}, speedup: {}, theoretical max speedup: {}'.format(single_network_duration, multi_network_duration, single_network_duration / multi_network_duration, num_networks))

    
if __name__ == '__main__':
    orig_nerf_vs_our_nerf()
