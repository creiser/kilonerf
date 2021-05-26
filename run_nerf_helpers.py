import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kilonerf_cuda
import math

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
    

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, direction_layer_size=None, use_initialization_fix=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.use_initialization_fix = use_initialization_fix
        
        if direction_layer_size is None:
            direction_layer_size = W//2
            
        def linear_layer(in_features, out_features, activation):
            if self.use_initialization_fix:
                return DenseLayer(in_features, out_features, activation=activation)
            else:
                return nn.Linear(in_features, out_features)
            
        self.pts_linears = nn.ModuleList(
            [linear_layer(input_ch, W, activation="relu")] + [linear_layer(W, W, activation="relu") if i not in self.skips else linear_layer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([linear_layer(input_ch_views + W, direction_layer_size, activation="relu")])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
     
        if use_viewdirs:
            self.feature_linear = linear_layer(W, W, activation="linear")
            self.alpha_linear = linear_layer(W, 1, activation="linear")
            self.rgb_linear = linear_layer(direction_layer_size, 3, activation="linear")
        else:
            self.output_linear = linear_layer(W, output_ch, activation="linear")

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
        
class CoarseAndFine(nn.Module):
    def __init__(self, model_coarse, model_fine) :
        super(CoarseAndFine, self).__init__()
        self.model_coarse = model_coarse
        self.model_fine = model_fine

def replace_transparency_by_background_color(acc_map, background_color=None):
    res = 1. - acc_map[...,None]
    if background_color is not None:
        res = res * background_color
    return res

# Ray helpers
#def get_rays(H, W, focal, c2w):
def get_rays(intrinsics, c2w, expand_origin=True):
    root_num_blocks = 64 # => 4096 blocks
    root_num_threads = 16 # => 256 threads per block
    rays_d = kilonerf_cuda.get_rays_d(intrinsics.H, intrinsics.W, intrinsics.cx, intrinsics.cy, intrinsics.fx, intrinsics.fy, c2w[:3, :3].contiguous(), root_num_blocks, root_num_threads)

    '''
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - intrinsics.cx) / intrinsics.fx, -(j - intrinsics.cy) / intrinsics.fy, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    ''' 
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    if expand_origin:
        rays_o = rays_o.expand(rays_d.shape)
    else:
        rays_o = rays_o.contiguous()
    return rays_o, rays_d


#def get_rays_np(H, W, focal, c2w):
def get_rays_np(intrinsics, c2w):
    W, H = intrinsics.W, intrinsics.H

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - intrinsics.cx) / intrinsics.fx, -(j - intrinsics.cy) / intrinsics.fy, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

class ChainEmbeddingAndModel(nn.Module):
    def __init__(self, model, embed_fn, embeddirs_fn):
        super(ChainEmbeddingAndModel, self).__init__()
        self.model = model
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
    
    def forward(self, points_and_dirs):
        embedded_points = self.embed_fn(points_and_dirs[:, :3])
        if self.embeddirs_fn is not None:
            embedded_dirs = self.embeddirs_fn(points_and_dirs[:, 3:])
            embedded_points_and_dirs = torch.cat([embedded_points, embedded_dirs], -1)
            return self.model(embedded_points_and_dirs)
        else:
            return self.model(embedded_points)

def lookat(look_from, look_to, tmp = np.asarray([0., 0., 1.])):
    forward = look_from - look_to
    forward = forward / np.linalg.norm(forward)
    right = np.cross(tmp, forward)
    right = right / np.linalg.norm(right) # TODO: handle np.linalg.norm(right) == 0
    up = np.cross(forward, right)
    
    c2w_T = np.zeros((4,4))
    c2w_T[0,0:3] = right
    c2w_T[1,0:3] = up
    c2w_T[2,0:3] = forward
    c2w_T[3,0:3] = look_from
    c2w_T[3,3] = 1
    
    return c2w_T.T

class OrbitCamera:
    def __init__(self, center, radius, inclination, azimuth, device):
        self.center = center
        self.radius = radius
        self.inclination = inclination
        self.azimuth = azimuth
        self.device = device
        self.compute_c2w()
        
    def zoom(self, delta):
        self.radius += delta
        self.compute_c2w()
        
    def pan(self, delta_x, delta_y):
        c2w_T = self.c2w.cpu().numpy().T
        right = c2w_T[0,0:3]
        up    = c2w_T[1,0:3]
        self.center += delta_x * right
        self.center += delta_y * up
        self.compute_c2w()
    
    def rotate(self, delta_x, delta_y):
        self.azimuth += delta_x
        self.inclination += delta_y
        eps = 0.001
        self.inclination = min(max(eps, self.inclination), math.pi - eps)
        self.compute_c2w()
        
    def compute_c2w(self):
        offset = np.asarray([self.radius * math.cos(self.azimuth) * math.sin(self.inclination),
                             self.radius * math.sin(self.azimuth) * math.sin(self.inclination),
                             self.radius * math.cos(self.inclination)])
        look_from = self.center + offset
        look_to = self.center
        self.c2w = torch.tensor(lookat(look_from, look_to), dtype=torch.float, device=self.device)