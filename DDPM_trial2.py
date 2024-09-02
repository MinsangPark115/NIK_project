import os
import json
import argparse
import math
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
import torchvision.datasets
from torch.utils.data import Subset
from tqdm import tqdm
from PIL import Image
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.strategies import DDPStrategy
import easydict
import pickle
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import habana_frameworks.torch.gpu_migration
import habana_frameworks.torch.core as htcore

# if os.getenv('DEBUG', '0') == '1':
# os.environ['PT_HPU_LAZY_MODE'] = '1'
# os.environ['LOG_LEVEL_PT_FALLBACK'] = '1'
# os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '1'
# os.environ['LOG_LEVEL_ALL'] = '3'
# os.environ['ENABLE_CONSOLE'] = 'true'

print(torch.__version__) # 1.9.0+cu1x1 ??
# print(pl.__version__) # 0.8.5 ??

def make_beta_schedule(schedule, start, end, n_timestep):
     
    if schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'linear':
        betas = torch.linspace(start, end, n_timestep, dtype=torch.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(start, end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(start, end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = end * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / (torch.linspace(n_timestep, 1, n_timestep, dtype=torch.float64))
    else:
        raise NotImplementedError(schedule)

    return betas


def _warmup_beta(start, end, n_timestep, warmup_frac):
    
    betas               = end * torch.ones(n_timestep, dtype=torch.float64)
    warmup_time         = int(n_timestep * warmup_frac)
    betas[:warmup_time] = torch.linspace(start, end, warmup_time, dtype=torch.float64)

    return betas


def normal_kl(mean1, logvar1, mean2, logvar2):
    
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

    return kl


def extract(input, t, shape):
    out     = torch.gather(input, 0, t.to(input.device))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out     = out.reshape(*reshape)
    return out


def noise_like(shape, noise_fn, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])

        return noise_fn(*shape_one).repeat(shape[0], *resid)

    else:
        return noise_fn(*shape)


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    centered_x = x - means
    inv_stdv   = torch.exp(-log_scales)
    plus_in    = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus   = approx_standard_normal_cdf(plus_in)
    min_in     = inv_stdv * (centered_x - 1. / 255.)
    cdf_min    = approx_standard_normal_cdf(min_in)

    log_cdf_plus          = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1 - cdf_min, min=1e-12))
    cdf_delta             = cdf_plus - cdf_min
    log_probs             = torch.where(x < -0.999, log_cdf_plus,
                                        torch.where(x > 0.999, log_one_minus_cdf_min,
                                                    torch.log(torch.clamp(cdf_delta, min=1e-12))))

    return log_probs

class GaussianDiffusion(nn.Module):
    def __init__(self, betas, model_mean_type, model_var_type, loss_type):
        """
        input:
            betas (tensor) : 미리 정해진 forward process 의 variance schedule, 1- (size = (n_timesteps) )
            model_mean_type (str) : 논문 상에서는 eps 사용
            model_var_type (str) : fixedsmall, fixedlarge가 각각 논문 3.2장에서 나온 두 종류의 variance. 
            loss_type (str) : 논문 상에서는 kl 사용
        class 변수:
            self.register로 만들어진 변수들 : betas를 통해 계산할 수 있는 값들. 추후 likelihood 계산에 바로 사용하기 위해 미리 계산
        """
        super().__init__()

        betas              = betas.type(torch.float64)
        timesteps          = betas.shape[0]
        self.num_timesteps = int(timesteps)

        self.model_mean_type = model_mean_type  # xprev, xstart, eps
        self.model_var_type  = model_var_type   # learned, fixedsmall, fixedlarge
        self.loss_type       = loss_type        # kl, mse

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped",
                      torch.log(torch.cat((posterior_variance[1].view(1, 1),
                                           posterior_variance[1:].view(-1, 1)), 0)).view(-1))
        self.register("posterior_mean_coef1", (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)))
        self.register("posterior_mean_coef2", ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)))

    def register(self, name, tensor):
        """
        class 변수 등록을 위한 함수, class 선언 시에 변수로 설정된다.
        input:
            name (str) : 등록할 이름
            tensor (tensor) : 등록할 tensor
        """
        self.register_buffer(name, tensor.type(torch.float32))

        
        
        
    # ========================================================== forward process ==========================================================
    
    
    
    
    def q_mean_variance(self, x_0, t):
        """
        q(x_t|x_0)의 mean, variance, log_variance를 return하는 함수, 논문 (4)식 참조
        input:
            x_0 (tensor): input image batch (Size = (Batch 내 이미지 수, Channel 수, Width, Height) , CIFAR10의 Training Set 기준 (128,3,32,32))
        output:
            mean, variance, log_variance : t 시점의 q의 mean, variance, log-variance (size는 x_0와 같음)
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = extract(1. - self.alphas_cumprod, t, x_0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):
        """
        q(x_t|x_0) sampling 하는 함수, 논문 (4)식 참조
        input:
            x_0 (tensor): input image batch 
        output:
            t 시점의 q(x_t|x_0)을 sampling 한 tensor 
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)
    
    def q_sample_loop(self, x_0, T, device, freq=50):
        """
        q(x_t|x_0)들의 list를 return
        t는 0~T 까지 return할 수 있으며 freq에 따라 return list의 size가 변경된다.
        input:
            freq (int): freq step마다 imglist에 append해준다. 예를 들어 freq가 50이면 x_0, x_50, x_100...을 imglist에 넣어준다
        output:
            imglist (list of tensor)
        """
        noise = torch.randn_like(x_0)
        imglist = [x_0]

        clip = lambda x_: (x_.clamp(min=-1, max=1) )
        for i in range(T+1):
            if (i+1)%freq == 0:
                imglist.append(clip(self.q_sample(x_0.to(device), torch.full((128,) , i).to(device) , noise.to(device))))
        return imglist
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        q(x_(t-1)|x_t,x_0) 의 mean, variance, log_variance return하는 함수, 논문 (6), (7)식 참조
        input:
            x_0 (tensor): input image batch 
            x_t (tensor): forward process를 timestep t만큼 했을 때
        output:
            mean, var, log_var_clipped : mean, variance, log-variance of q(x_(t-1)|x_t,x_0) 
        """
        mean            = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
                           + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var             = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var_clipped

    
    
    
    # ========================================================== reverse process ==========================================================
    
    
    
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        mean type이 eps인 경우 x_0 예측, (12)식 변형
        """
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_start_from_prev(self, x_t, t, x_prev):
        """
        mean type이 xprev인 경우 x_0 예측, (7)식 변형    
        """

        return (extract(1./self.posterior_mean_coef1, t, x_t.shape) * x_prev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)
    
    def p_mean_variance(self, model, x, t, clip_denoised, return_pred_x0):
        """
        reverse step model의 mean, var type에 따라 다른 mean, variance, log_variance를 return하는 함수.
        var type이 fixedsmall, fixedlarge가 각각 논문 3.2장에서 나온 두 종류의 variance. 
        mean type이 eps인 경우가 논문의 (12)식 변형, xprev인 경우가 논문의 (7)식 변경
        input:
            model (class) : 본 논문에서는 Unet model
            cleap_denoised (bool) : clipping 여부 결정 
            return_pred_x0 (bool) : 예측한 x0 return 여부 결정 
            
        output:
            mean, var, log_var, pred_x0 : mean, variance, log_variance, predicted x0 of reverse process
        """

        model_output = model(x, t)

        # Learned or fixed variance?
        if self.model_var_type == 'learned':
            model_output, log_var = torch.split(model_output, 2, dim=-1)
            var                   = torch.exp(log_var)

        elif self.model_var_type in ['fixedsmall', 'fixedlarge']:

            # below: only log_variance is used in the KL computations
            var, log_var = {
                # for 'fixedlarge', we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas, torch.log(torch.cat((self.posterior_variance[1].view(1, 1),
                                                                self.betas[1:].view(-1, 1)), 0)).view(-1)),
                'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.model_var_type]

            var     = extract(var, t, x.shape) * torch.ones_like(x)
            log_var = extract(log_var, t, x.shape) * torch.ones_like(x)
        else:
            raise NotImplementedError(self.model_var_type)

        # Mean parameterization
        _maybe_clip = lambda x_: (x_.clamp(min=-1, max=1) if clip_denoised else x_)

        if self.model_mean_type == 'xprev':
            # the model predicts x_{t-1}
            pred_x_0 = _maybe_clip(self.predict_start_from_prev(x_t=x, t=t, x_prev=model_output))
            mean     = model_output
        elif self.model_mean_type == 'xstart':
            # the model predicts x_0
            pred_x0    = _maybe_clip(model_output)
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        elif self.model_mean_type == 'eps':
            # the model predicts epsilon
            pred_x0    = _maybe_clip(self.predict_start_from_noise(x_t=x, t=t, noise=model_output))
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred_x0:
            return mean, var, log_var, pred_x0
        else:
            return mean, var, log_var



    def p_sample(self, model, x, t, noise_fn, clip_denoised=True, return_pred_x0=False):
        """
        t가 T...1일 때 mean, log_variance 이용해 x_(t-1) sampling, 논문의 Algorithm 2 참조
        input:
            noise_fn : 논문 상에서는 z로 표기
        output:
            sample : x_(t-1)
        """

        mean, _, log_var, pred_x0 = self.p_mean_variance(model, x, t, clip_denoised, return_pred_x0=True)
        noise                     = noise_fn(x.shape, dtype=x.dtype).to(x.device)

        shape        = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape).to(x.device)
        sample       = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        return (sample, pred_x0) if return_pred_x0 else sample

    @torch.no_grad()
    def p_sample_loop(self, model, shape, noise_fn=torch.randn):
        """
        t=T부터 0까지 이전 생성 image 이용해 sampling. t=0때 sampling 완료 후 image tensor return한다. 논문의 Algorithm 2 구현
        input:
            shape (tuple) : image tensor size 
        output:
            img (tensor) : x_0에 해당하는 image tensor 
        """

        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        img    = noise_fn(shape).to(device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
                return_pred_x0=False
            )

        return img

    @torch.no_grad()
    def p_sample_loop_progressive(self, model, shape, device, noise_fn=torch.randn, include_x0_pred_freq=50, ret_type="img"):
        """
        p_sample_loop와 유사. 
        input:
            include_x0_pred_freq (int) : include_x0_pred_freq 번 마다 그 시점의 x_0 예측값을 기록
            ret_type (string): img 면 image 하나를 return, list면 generation process의 모든 timestep의 image list를 return. 
        output:
            x0_preds_ (tensor) : (B, num_recorded_x0_pred, C, H, W) size의 tensor
            
        """

        img = noise_fn(shape, dtype=torch.float32).to(device)

        num_recorded_x0_pred = self.num_timesteps // include_x0_pred_freq
        x0_preds_            = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32).to(device)
        
        
        clip = lambda x_: (x_.clamp(min=-1, max=1))
        imglist = [clip(img)]
        
        predx0_list = []
        for i in tqdm(reversed(range(self.num_timesteps))):

            # Sample p(x_{t-1} | x_t) as usual
            img, pred_x0 = self.p_sample(model=model,
                                         x=img,
                                         t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
                                         noise_fn=noise_fn,
                                         return_pred_x0=True)
            imglist.append(clip(img))

            # Keep track of prediction of x0
            insert_mask = np.floor(i // include_x0_pred_freq) == torch.arange(num_recorded_x0_pred,
                                                                              dtype=torch.int32,
                                                                              device=device)

            insert_mask = insert_mask.to(torch.float32).view(1, num_recorded_x0_pred, *([1] * len(shape[1:])))
            x0_preds_   = insert_mask * pred_x0[:, None, ...] + (1. - insert_mask) * x0_preds_
            if i % include_x0_pred_freq == 0:
                predx0_list.append(pred_x0)
        if ret_type == "list":
            return imglist, predx0_list
        elif ret_type == "img":
            return img, x0_preds_

        
        
        
        
    # ========================================================== Log likelihood calculation ==========================================================

    
    
    
    
    def _vb_terms_bpd(self, model, x_0, x_t, t, clip_denoised, return_pred_x0):
        """
        논문에서 사용한 loss를 구하는 함수. (5)식 참조
        t가 0인 image는 decoder의 negative log likelihood 값 return
        t가 0이 아닌 image는 KL-Divergence return    
        """

        batch_size = t.shape[0]
        true_mean, _, true_log_variance_clipped    = self.q_posterior_mean_variance(x_0=x_0,
                                                                                    x_t=x_t,
                                                                                    t=t)
        model_mean, _, model_log_variance, pred_x0 = self.p_mean_variance(model,
                                                                          x=x_t,
                                                                          t=t,
                                                                          clip_denoised=clip_denoised,
                                                                          return_pred_x0=True)

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)

        decoder_nll = -discretized_gaussian_log_likelihood(x_0, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = torch.mean(decoder_nll.view(batch_size, -1), dim=1) / np.log(2.)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where(t == 0, decoder_nll, kl)

        return (output, pred_x0) if return_pred_x0 else output

    def training_losses(self, model, x_0, t, noise=None):
        """
        loss type이 kl인 경우 위의 _vb_terms_bpd 값 return
        mse인 경우 mse loss return    
        """

        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)

        
        # Calculate the loss
        if self.loss_type == 'kl':
            # the variational bound
            losses = self._vb_terms_bpd(model=model, x_0=x_0, x_t=x_t, t=t, clip_denoised=False, return_pred_x0=False)

        elif self.loss_type == 'mse':
            # unweighted MSE
            assert self.model_var_type != 'learned'
            target = {
                'xprev': self.q_posterior_mean_variance(x_0=x_0, x_t=x_t, t=t)[0],
                'xstart': x_0,
                'eps': noise
            }[self.model_mean_type]

            model_output = model(x_t, t)
            losses       = torch.mean((target - model_output).view(x_0.shape[0], -1)**2, dim=1)

        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def _prior_bpd(self, x_0):
        """
        x_0의 prior distribution에 대한 복원 오차 계산하는 함수
        """

        B, T                        = x_0.shape[0], self.num_timesteps
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0,
                                                           t=torch.full((B,), T - 1, dtype=torch.int64))
        kl_prior                    = normal_kl(mean1=qt_mean,
                                                logvar1=qt_log_variance,
                                                mean2=torch.zeros_like(qt_mean),
                                                logvar2=torch.zeros_like(qt_log_variance))

        return torch.mean(kl_prior.view(B, -1), dim=1)/np.log(2.)

    @torch.no_grad()
    def calc_bpd_loop(self, model, x_0, clip_denoised):
        """
        모든 timestep 에 대한 prior distribution에 대한 복원 오차 계산하는 함수    
        """

        (B, C, H, W), T = x_0.shape, self.num_timesteps

        new_vals_bt = torch.zeros((B, T))
        new_mse_bt  = torch.zeros((B, T))

        for t in reversed(range(self.num_timesteps)):

            t_b = torch.full((B, ), t, dtype=torch.int64)

            # Calculate VLB term at the current timestep
            new_vals_b, pred_x0 = self._vb_terms_bpd(model=model,
                                                     x_0=x_0,
                                                     x_t=self.q_sample(x_0=x_0, t=t_b),
                                                     t=t_b,
                                                     clip_denoised=clip_denoised,
                                                     return_pred_x0=True)

            # MSE for progressive prediction loss
            new_mse_b = torch.mean((pred_x0-x_0).view(B, -1)**2, dim=1)

            # Insert the calculated term into the tensor of all terms
            mask_bt = (t_b[:, None] == torch.arange(T)[None, :]).to(torch.float32)

            new_vals_bt = new_vals_bt * (1. - mask_bt) + new_vals_b[:, None] * mask_bt
            new_mse_bt  = new_mse_bt  * (1. - mask_bt) + new_mse_b[:, None] * mask_bt

        prior_bpd_b = self._prior_bpd(x_0)
        total_bpd_b = torch.sum(new_vals_bt, dim=1) + prior_bpd_b

        return total_bpd_b, new_vals_bt, prior_bpd_b, new_mse_bt    


## Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

def swish(input):
    
    return input * torch.sigmoid(input)


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    scale=1,
    mode="fan_avg",
):
    
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Swish(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, time_dim, dropout):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(Swish(), linear(time_dim, out_channel))

        self.norm2 = nn.GroupNorm(32, out_channel)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        out = out + self.time(time).view(batch, -1, 1, 1)

        out = self.conv2(self.dropout(self.activation2(self.norm2(out))))

        if self.skip is not None:
            input = self.skip(input)

        return out + input


class SelfAttention(nn.Module):
    
    def __init__(self, in_channel):
        super().__init__()

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 4, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape

        norm = self.norm(input)
        qkv = self.qkv(norm)
        query, key, value = qkv.chunk(3, dim=1)

        attn = torch.einsum("nchw, ncyx -> nhwyx", query, key).contiguous() / math.sqrt(
            channel
        )
        attn = attn.view(batch, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, height, width, height, width)

        out = torch.einsum("nhwyx, ncyx -> nchw", attn, input).contiguous()
        out = self.out(out)

        return out + input


class TimeEmbedding(nn.Module):
    
    def __init__(self, dim):
        super().__init__()

        self.dim      = dim
        half_dim      = self.dim // 2
        self.inv_freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000) / (half_dim - 1)))

    def forward(self, input):
        shape       = input.shape
        input       = input.view(-1).to(torch.float32)
        sinusoid_in = torch.ger(input, self.inv_freq.to(input.device))
        pos_emb     = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb     = pos_emb.view(*shape, self.dim)
        
        return pos_emb


class ResBlockWithAttention(nn.Module):
    
    def __init__(self, in_channel, out_channel, time_dim, dropout, use_attention=False):
        super().__init__()

        self.resblocks = ResBlock(in_channel, out_channel, time_dim, dropout)

        if use_attention:
            self.attention = SelfAttention(out_channel)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


def spatial_fold(input, fold):
    
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold):
    
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )


class UNet(nn.Module):
    
    def __init__(
        self,
        in_channel,
        channel,
        channel_multiplier,
        n_res_blocks,
        attn_strides,
        dropout=0,
        fold=1,
    ):
        super().__init__()

        self.fold = fold

        time_dim = channel * 4

        n_block = len(channel_multiplier)

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        down_layers   = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel    = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                ),
                ResBlockWithAttention(
                    in_channel, in_channel, time_dim, dropout=dropout
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                in_channel = channel_mult

            if i != 0:
                up_layers.append(Upsample(in_channel))

        self.up = nn.ModuleList(up_layers)

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            conv2d(in_channel, 3 * (fold ** 2), 3, padding=1, scale=1e-10),
        )

    def forward(self, input, time):
        time_embed = self.time(time)
        
        feats = []

        out = spatial_fold(input, self.fold)
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)

            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)

            else:
                out = layer(out)

        out = self.out(out)
        out = spatial_unfold(out, self.fold)

        return out


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def get_train_data(conf):
    if conf.dataset.name == 'cifar10':
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(conf.dataset.path,
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
        valid_set = torchvision.datasets.CIFAR10(conf.dataset.path,
                                                  train=True,
                                                  transform=transform_test,
                                                  download=True)

        num_train  = len(train_set)
        indices    = torch.randperm(num_train).tolist()
        valid_size = int(np.floor(0.05 * num_train))

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        train_set = Subset(train_set, train_idx)
        valid_set = Subset(valid_set, valid_idx)

    

    else:
        raise FileNotFoundError

    return train_set, valid_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DDP:
    def __init__(self, conf):
        self.conf = conf

        # 모델 초기화
        self.model = UNet(
            self.conf.model.in_channel,
            self.conf.model.channel,
            channel_multiplier=self.conf.model.channel_multiplier,
            n_res_blocks=self.conf.model.n_res_blocks,
            attn_strides=self.conf.model.attn_strides,
            dropout=self.conf.model.dropout,
            fold=self.conf.model.fold,
        )
        self.ema = UNet(
            self.conf.model.in_channel,
            self.conf.model.channel,
            channel_multiplier=self.conf.model.channel_multiplier,
            n_res_blocks=self.conf.model.n_res_blocks,
            attn_strides=self.conf.model.attn_strides,
            dropout=self.conf.model.dropout,
            fold=self.conf.model.fold,
        )

        self.betas = make_beta_schedule(
            schedule=self.conf.model.schedule.type,
            start=self.conf.model.schedule.beta_start,
            end=self.conf.model.schedule.beta_end,
            n_timestep=self.conf.model.schedule.n_timestep,
        )

        self.diffusion = GaussianDiffusion(
            betas=self.betas,
            model_mean_type=self.conf.model.mean_type,
            model_var_type=self.conf.model.var_type,
            loss_type=self.conf.model.loss_type,
        )
        self.model.to(device)
        self.ema.to(device)

    def setup(self):
        # 데이터셋 로드
        self.train_set, self.valid_set = get_train_data(self.conf)

    def forward(self, x):
        return self.diffusion.p_sample_loop(self.model, x.shape)

    def configure_optimizers(self):
        if self.conf.training.optimizer.type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.conf.training.optimizer.lr)
        else:
            raise NotImplementedError("지원되지 않는 옵티마이저입니다.")
        return optimizer

    def training_step(self, batch):
        img, _ = batch
        img=img.to(device)
        time = (torch.rand(img.shape[0]) * 1000).type(torch.int64).to(img.device)
        loss = self.diffusion.training_losses(self.model, img, time).mean()

        accumulate(self.ema, self.model, 0.9999)

        return loss

    def validation_step(self, batch):
        img, _ = batch
        img = img.to(device)
        time = (torch.rand(img.shape[0]) * 1000).type(torch.int64).to(img.device)
        loss = self.diffusion.training_losses(self.ema, img, time).mean()

        return loss

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.conf.training.dataloader.batch_size,
            shuffle=True,
            num_workers=self.conf.training.dataloader.num_workers,
            pin_memory=True,
            drop_last=self.conf.training.dataloader.drop_last,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_set,
            batch_size=self.conf.validation.dataloader.batch_size,
            shuffle=False,
            num_workers=self.conf.validation.dataloader.num_workers,
            pin_memory=True,
            drop_last=self.conf.validation.dataloader.drop_last,
        )
        return valid_loader

    def train(self):
        optimizer = self.configure_optimizers()
        train_loader = self.train_dataloader()
        val_loader = self.val_dataloader()
        device = next(self.model.parameters()).device
        print(f"Model is running on: {device}")

        for epoch in tqdm(range(self.conf.training.epochs)):
            self.model.train()
            for batch in tqdm(train_loader, desc="Training", leave=False):
                optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                htcore.mark_step()
                optimizer.step()
                htcore.mark_step()

            # 검증 단계
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for batch in val_loader:
                    loss = self.validation_step(batch)
                    val_losses.append(loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Epoch {epoch}, Validation Loss: {avg_val_loss}')

            # 샘플 생성 및 로그 기록
            # if epoch % self.conf.training.sample_freq == 0:
            #     print("sampling...")
            #     self.sample_images(epoch)

            # 모델 저장
            if epoch % self.conf.training.ckpt_freq == 0:
                print("saving...")
                self.save_checkpoint(epoch, avg_val_loss)

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_dir = os.path.join(self.conf.ckpt_dir, f'ddp_{epoch:02d}-{val_loss:.2f}.pt')
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict()}, checkpoint_dir)
        print(f'Checkpoint saved at {checkpoint_dir}')

    def sample_images(self, epoch):
        shape = (16, 3, self.conf.dataset.resolution, self.conf.dataset.resolution)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sample = progressive_samples_fn(self.ema, self.diffusion, shape, device=device)
        print(sample['samples'].shape)
        # 샘플 이미지 저장
        grid = make_grid(sample['samples'], nrow=4)
        print(grid.shape)
        print(self.conf.sample_dir)
        
        save_image_pil(grid, os.path.join(self.conf.sample_dir, f'generated_images_{epoch}.png'))

        # save_image(grid, os.path.join(self.conf.sample_dir, f'generated_images_{epoch}.png'))
        print(self.sample['progressive_samples'].reshape(-1, 3, self.conf.dataset.resolution, self.conf.dataset.resolution).shape)
        grid_progressive = make_grid(sample['progressive_samples'].reshape(-1, 3, self.conf.dataset.resolution, self.conf.dataset.resolution), nrow=20)
        
        save_image_pil(grid_progressive, os.path.join(self.conf.sample_dir, f'progressive_generated_images_{epoch}.png'))

        # save_image(grid, os.path.join(self.conf.sample_dir, f'progressive_generated_images_{epoch}.png'))

def save_image_pil(tensor, file_path):
    tensor = tensor.clone()
    print("here1")
    tensor = tensor.to("cpu")
    print("here1.5")
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        print("here2")
    
    tensor = tensor * 255
    print("here3")
    tensor = tensor.byte()
    print("here4")
    np_image = tensor.numpy().transpose(1, 2, 0)
    print("here5")
    
    image = Image.fromarray(np_image)
    print("here6")
    image.save(file_path)

class obj(object):
    """
    config가 json 형식으로 되어있습니다.
    json 파일의 데이터를 받아오는 형식을 바꿔주는 함수입니다. 
    """
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)


def accumulate(model1, model2, decay=0.9999):
    """
    아래 DDP class에서 Exponential Moving Average를 진행하기 위한 함수입니다.
    model1의 parameter을 기존 param*decay + model2의 param*(1-decay)로 update 합니다.
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def samples_fn(model, diffusion, shape):
    """
    E(x_(t-1)|x_(t)) 을 이미지 화 하기 위한 함수
    sample image의 현재 범위는 -1~1입니다.
    이를 image화 시키려먼 0~1 사이의 범위로 만들어주어야 하고, 이를 수행하는 함수입니다.
    """
    samples = diffusion.p_sample_loop(model=model,
                                      shape=shape,
                                      noise_fn=torch.randn)
    return {
      'samples': (samples + 1)/2
    }


def progressive_samples_fn(model, diffusion, shape, device, include_x0_pred_freq=50, ret_type = "img"):
    """
    위 samples_fn과 역할이 동일합니다. 다만 차이점은 image로 만드는 것이 x_(t-1)|x_(t) 뿐 아니라 x_(0)|x_(t)인 progressive sample도 있다는 점입니다.
    """
    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        ret_type = ret_type
    )
    if ret_type == "list":
        for i in range(len(samples)):
            samples[i] = (samples[i] + 1)/2
        for j in range(len(progressive_samples)):
            progressive_samples[j] = (progressive_samples[j]+1)/2
        return {'samples': samples, 'progressive_samples': progressive_samples}
    elif ret_type == "img":
        samples = (samples +1)/2
    return {'samples': samples, 'progressive_samples': (progressive_samples + 1)/2}


def bpd_fn(model, diffusion, x):
    """
    복원 오차 계산하는 함수. diffusion.calc_bpd_loop 참고
    """
    total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = diffusion.calc_bpd_loop(model=model, x_0=x, clip_denoised=True)

    return {
      'total_bpd': total_bpd_b,
      'terms_bpd': terms_bpd_bt,
      'prior_bpd': prior_bpd_b,
      'mse': mse_bt
    }


def validate(val_loader, model, diffusion):
    """
    validation dataset에 대해 bpd, mse 계산하는 함수
    """
    model.eval()
    bpd = []
    mse = []
    with torch.no_grad():
        for i, (x, y) in enumerate(iter(val_loader)):
            x       = x
            metrics = bpd_fn(model, diffusion, x)

            bpd.append(metrics['total_bpd'].view(-1, 1))
            mse.append(metrics['mse'].view(-1, 1))

        bpd = torch.cat(bpd, dim=0).mean()
        mse = torch.cat(mse, dim=0).mean()

    return bpd, mse

args = easydict.EasyDict({
        "train": False,
        "config": 'config/diffusion_cifar10.json',
        "ckpt_dir": 'ckpts',
        "ckpt_freq": 5,
        "n_gpu": 1,
        "model_dir": 'ckpts/last.ckpt',
        "sample_dir": 'samples',
        "prog_sample_freq": 200,
        "n_samples": 100
    })

    

path_to_config = args.config
with open(path_to_config, 'r') as f:
    conf = json.load(f)

conf = obj(conf)
denoising_diffusion_model = DDP(conf)




denoising_diffusion_model.setup()
denoising_diffusion_model.train()