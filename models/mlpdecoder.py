# Source from NeuralLift-360 (https://github.com/VITA-Group/NeuralLift-360/blob/main/nerf/network.py)

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from nerf.renderer import NeRFRenderer

import numpy as np

from nerf.general_utils import safe_normalize, sample_pdf

import raymarching


# TODO: not sure about the details...
class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU(inplace=True)

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out    

class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.ModuleList(net)
        
    
    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x


class MLPDecoder(NeRFRenderer):
    def __init__(self,
        opt,
        hidden_dim=128, # 128 in paper
        num_sigma_layers=2, # Additional layers for density only
        num_bg_layers=2
    ):
        super().__init__(opt)
        
        self.input_triplane_dim = 32
        self.input_view_dim = 3
        
        self.num_sigma_layers = num_sigma_layers
        self.num_bg_layers = num_bg_layers
        
        self.fourier_transform = FourierFeatureTransform(32, 64, scale=1)
        self.common_net = MLP(self.input_triplane_dim, hidden_dim, hidden_dim, 2, bias=True)
        self.sigma_net = MLP(hidden_dim, 1, hidden_dim, 1, bias=True)
        self.bg_net = MLP(hidden_dim + self.input_view_dim, 3, hidden_dim, num_bg_layers+1, bias=True)

        self.smooth_reg_p_dist = 0.004
    
    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = F.grid_sample(plane,
                                         coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                         mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features
        
    
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        # g = self.blob_density * torch.exp(- d / (self.blob_radius ** 2))
        g = self.blob_density * (1 - torch.sqrt(d) / self.blob_radius)

        return g
    
    
    def density(self, x, triplane_feat):
        # x : [B, N, 3] in [-bound, bound]
        # triplane_feat : [B, 256, 256, 32, 3]
        
        xy_embed = self.sample_plane(x[..., 0:2], triplane_feat[0])
        yz_embed = self.sample_plane(x[..., 1:3], triplane_feat[1])
        xz_embed = self.sample_plane(x[..., :3:2], triplane_feat[2])
        
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 
        feat = self.fourier_transform(features)
        feat = self.common_net(feat)
        out = self.sigma_net(feat)
        
        sigma = F.softplus(out)  # F.softplus(feat[..., 0] + self.density_blob(x))
        
        return {
            'sigma': sigma, 
            'feat': feat
        }
    
    
    def color(self, d, features):
        # d : [B, N, 3], view direction, normalized in [-1, 1]
        # features : [B, N, D] mlp intermediate feature
        
        out = self.bg_net(features, d)
        color = torch.sigmoid(out)
        
        return color
    
    
    def forward(self, x, d, triplane_feat, l=None, ratio=1, shading='albedo', l_p=None, l_a=None):
        # x : [N, 3] in [-bound, bound]
        # d : [N, 3], view direction, normalized in [-1, 1]
        # triplane_feat : [256, 256, 32, 3]
        
        if shading == 'albedo':
            density_output = self.density(x, triplane_feat)
            sigma = density_output['sigma']
            color = self.color(d, density_output['feat'])
            normal = None
        else:
            with torch.enable_grad():
                x.requires_grad_(True)
                density_output = self.density(x, triplane_feat)
                sigma = density_output['sigma']
                albedo = self.color(d, density_output['feat'])
                # query gradient
                normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
            normal = safe_normalize(normal)
            # normal = torch.nan_to_num(normal)
            # normal = normal.detach()

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
        
        # normal
        
        return sigma, color, normal

        
    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0


    def run(self, triplane_feat, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        # triplane_feat : [B, 256, 256, 32, 3]
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(rays_o[0] + torch.randn(3, device=device, dtype=torch.float) * 0.1)
            light_d = light_d * (random.random() * 0.8 + 0.7)
            
        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3), triplane_feat)

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                #alphas = 1 - torch.exp(-deltas * sigmas.squeeze(-1)) # [N, T]
                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3), triplane_feat)
            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            tmp_output = torch.cat([sigmas, new_density_outputs['sigma']], dim=1)
            #sigmas = torch.gather(tmp_output, dim=1, index=z_index.unsquueze(-1).expand_as(tmp_output))
            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        #alphas = 1 - torch.exp(-deltas * sigmas.squeeze(-1)) # [N, T+t]
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]
        results['weights'] = weights

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        #sigmas = sigmas.view(-1, sigmas.shape[-1])
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), triplane_feat)
        results['sigmas'] = sigmas
        
        # smoothness regularization loss
        if self.training:
            xyzs_perturbed = xyzs + torch.randn_like(xyzs) * self.smooth_reg_p_dist
            results['sigmas_perturbed'] = self.density(xyzs_perturbed, triplane_feat)['sigma']        
        
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

        if normals is not None:
            # orientation loss
            normals = normals.view(N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.sum(-1).mean()
            # results['loss_orient'] = loss_orient.mean()

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears))
        # ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # For distortion loss
        midpoint = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
        midpoint = torch.cat([midpoint, (midpoint[..., -1:])], dim=-1)
        results['deltas'] = deltas
        results['midpoint'] = midpoint

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]
        if normals is not None:
            normals_im = torch.sum(weights.unsqueeze(-1) * normals, dim=-2) # [N, 3], in [0, 1]
            results['normals'] = normals_im.view(*prefix, 3)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            # sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 0
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        mask = (nears < fars).reshape(*prefix)

        sigmas = sigmas.unsqueeze(-1) # [M, 1]
        total_sigma = sigmas.sum(dim=0, keepdim=True) # [1, 1]
        # print(sigmas.shape, total_sigma.shape, xyzs.shape) # torch.Size([2097152, 1]) torch.Size([1, 1]) torch.Size([16384, 128, 3])
        results['origin'] = (xyzs.view(-1, 3) * sigmas / total_sigma).sum(dim=0) # [M, 3] --> [3]

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['bg_color'] = bg_color

        return results


    def run_cuda(self, triplane_feat, rays_o, rays_d, num_steps=128, dt_gamma=0, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # triplane_feat: [B, 256, 256, 32, 3]
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, dt_gamma, max_steps)

            #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            
            sigmas, rgbs, normals = self(xyzs, dirs, triplane_feat)
            results['sigmas'] = sigmas
        
            # smoothness regularization loss
            if self.training:
                xyzs_perturbed = xyzs + torch.randn_like(xyzs) * self.smooth_reg_p_dist
                results['sigmas_perturbed']  = self.density(xyzs_perturbed, triplane_feat)['sigma']
                
            z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
            z_vals = z_vals.expand((N, num_steps)) # [N, T]
            z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

            # perturb z_vals
            sample_dist = (fars - nears) / num_steps
            if perturb:
                z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
                
            deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
            deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
            
            # For distortion loss
            midpoint = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
            midpoint = torch.cat([midpoint, (midpoint[..., -1:])], dim=-1)
            results['deltas'] = deltas
            results['midpoint'] = midpoint

            #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh)

            # normals related regularizations
            if normals is not None:
                # orientation loss
                # weights = 1 - torch.exp(-sigmas)
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()

                # surface normal smoothness
                # normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                # loss_smooth = (normals - normals_perturb).abs()
                # results['loss_smooth'] = loss_smooth.mean()

            # weights normalization
            results['weights'] = weights

        else:
           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs, normals = self(xyzs, dirs, triplane_feat)

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]
                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # mix background color
        if self.bg_radius > 0:
            
            # use the bg model to calculate bg_color
            # sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(rays_d) # [N, 3]

        elif bg_color is None:
            bg_color = 0

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        # mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        # results['mask'] = mask
        results['bg_color'] = bg_color

        return results


    def render(self, triplane_feats, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # triplane_feats: [B, 256, 256, 32, 3]
        # rays_o, rays_d: [B, N, 3]
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            # _run = self.run_mixed
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            weights_sum = torch.empty((B, N), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(triplane_feats[b], rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    weights_sum[b:b+1, head:tail] = results_['weights_sum']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum

        else:
            results = _run(rays_o, rays_d, **kwargs)


        return results
    
    
    