# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import gen_models.utils.ops as ops
import gen_models.utils.misc as misc


MODEL = misc.make_empty_object()
# type of backbone architectures of the generator and discriminator \in
# ["deep_conv", "resnet", "big_resnet", "big_resnet_deep_legacy", "big_resnet_deep_studiogan", "stylegan2", "stylegan3"]
MODEL.backbone = "resnet"
# conditioning method of the generator \in ["W/O", "cBN", "cAdaIN"]
MODEL.g_cond_mtd = "cBN"
# conditioning method of the discriminator \in ["W/O", "AC", "PD", "MH", "MD", "2C","D2DCE", "SPD"]
MODEL.d_cond_mtd = "W/O"
# type of auxiliary classifier \in ["W/O", "TAC", "ADC"]
MODEL.aux_cls_type = "W/O"
# whether to normalize feature maps from the discriminator or not
MODEL.normalize_d_embed = False
# dimension of feature maps from the discriminator
# only appliable when MODEL.d_cond_mtd \in ["2C, D2DCE"]
MODEL.d_embed_dim = "N/A"
# whether to apply spectral normalization on the generator
MODEL.apply_g_sn = False
# whether to apply spectral normalization on the discriminator
MODEL.apply_d_sn = False
# type of activation function in the generator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
MODEL.g_act_fn = "ReLU"
# type of activation function in the discriminator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
MODEL.d_act_fn = "ReLU"
# whether to apply self-attention proposed by zhang et al. (SAGAN)
MODEL.apply_attn = False
# locations of the self-attention layer in the generator (should be list type)
MODEL.attn_g_loc = ["N/A"]
# locations of the self-attention layer in the discriminator (should be list type)
MODEL.attn_d_loc = ["N/A"]
# prior distribution for noise sampling \in ["gaussian", "uniform"]
MODEL.z_prior = "gaussian"
# dimension of noise vectors
MODEL.z_dim = 128
# dimension of intermediate latent (W) dimensionality used only for StyleGAN
MODEL.w_dim = "N/A"
# dimension of a shared latent embedding
MODEL.g_shared_dim = "N/A"
# base channel for the resnet style generator architecture
MODEL.g_conv_dim = 64
# base channel for the resnet style discriminator architecture
MODEL.d_conv_dim = 64
# generator's depth for "models/big_resnet_deep_*.py"
MODEL.g_depth = "N/A"
# discriminator's depth for "models/big_resnet_deep_*.py"
MODEL.d_depth = "N/A"
# whether to apply moving average update for the generator
MODEL.apply_g_ema = False
# decay rate for the ema generator
MODEL.g_ema_decay = "N/A"
# starting step for g_ema update
MODEL.g_ema_start = "N/A"
# weight initialization method for the generator \in ["ortho", "N02", "glorot", "xavier"]
MODEL.g_init = "ortho"
# weight initialization method for the discriminator \in ["ortho", "N02", "glorot", "xavier"]
MODEL.d_init = "ortho"
# type of information for infoGAN training \in ["N/A", "discrete", "continuous", "both"]
MODEL.info_type = "N/A"
# way to inject information into Generator \in ["N/A", "concat", "cBN"]
MODEL.g_info_injection = "N/A"
# number of discrete c to use in InfoGAN
MODEL.info_num_discrete_c = "N/A"
# number of continuous c to use in InfoGAN
MODEL.info_num_conti_c = "N/A"
# dimension of discrete c to use in InfoGAN (one-hot)
MODEL.info_dim_discrete_c = "N/A"


MODULES = misc.make_empty_object()
if MODEL.apply_g_sn:
    MODULES.g_conv2d = ops.snconv2d
    MODULES.g_deconv2d = ops.sndeconv2d
    MODULES.g_linear = ops.snlinear
    MODULES.g_embedding = ops.sn_embedding
else:
    MODULES.g_conv2d = ops.conv2d
    MODULES.g_deconv2d = ops.deconv2d
    MODULES.g_linear = ops.linear
    MODULES.g_embedding = ops.embedding

if MODEL.apply_d_sn:
    MODULES.d_conv2d = ops.snconv2d
    MODULES.d_deconv2d = ops.sndeconv2d
    MODULES.d_linear = ops.snlinear
    MODULES.d_embedding = ops.sn_embedding
else:
    MODULES.d_conv2d = ops.conv2d
    MODULES.d_deconv2d = ops.deconv2d
    MODULES.d_linear = ops.linear
    MODULES.d_embedding = ops.embedding

if MODEL.g_cond_mtd == "cBN" or MODEL.g_info_injection == "cBN" or MODEL.backbone == "big_resnet":
    MODULES.g_bn = ops.ConditionalBatchNorm2d
elif MODEL.g_cond_mtd == "W/O":
    MODULES.g_bn = ops.batchnorm_2d
elif MODEL.g_cond_mtd == "cAdaIN":
    pass
else:
    raise NotImplementedError

if not MODEL.apply_d_sn:
    MODULES.d_bn = ops.batchnorm_2d

if MODEL.g_act_fn == "ReLU":
    MODULES.g_act_fn = nn.ReLU(inplace=True)
elif MODEL.g_act_fn == "Leaky_ReLU":
    MODULES.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
elif MODEL.g_act_fn == "ELU":
    MODULES.g_act_fn = nn.ELU(alpha=1.0, inplace=True)
elif MODEL.g_act_fn == "GELU":
    MODULES.g_act_fn = nn.GELU()
elif MODEL.g_act_fn == "Auto":
    pass
else:
    raise NotImplementedError

if MODEL.d_act_fn == "ReLU":
    MODULES.d_act_fn = nn.ReLU(inplace=True)
elif MODEL.d_act_fn == "Leaky_ReLU":
    MODULES.d_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
elif MODEL.d_act_fn == "ELU":
    MODULES.d_act_fn = nn.ELU(alpha=1.0, inplace=True)
elif MODEL.d_act_fn == "GELU":
    MODULES.d_act_fn = nn.GELU()
elif MODEL.g_act_fn == "Auto":
    pass
else:
    raise NotImplementedError


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_cond_mtd, g_info_injection, affine_input_dim, MODULES):
        super(GenBlock, self).__init__()
        self.g_cond_mtd = g_cond_mtd
        self.g_info_injection = g_info_injection

        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            self.bn1 = MODULES.g_bn(in_features=in_channels)
            self.bn2 = MODULES.g_bn(in_features=out_channels)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            self.bn1 = MODULES.g_bn(affine_input_dim, in_channels, MODULES)
            self.bn2 = MODULES.g_bn(affine_input_dim, out_channels, MODULES)
        else:
            raise NotImplementedError

        self.activation = MODULES.g_act_fn
        self.conv2d0 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.g_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, affine):
        x0 = x
        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            x = self.bn1(x)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            x = self.bn1(x, affine)
        else:
            raise NotImplementedError
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        if self.g_cond_mtd == "W/O" and self.g_info_injection in ["N/A", "concat"]:
            x = self.bn2(x)
        elif self.g_cond_mtd == "cBN" or self.g_info_injection == "cBN":
            x = self.bn2(x, affine)
        else:
            raise NotImplementedError
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn, attn_g_loc, g_cond_mtd, num_classes,
                 g_init, g_depth, mixed_precision):
        super(Generator, self).__init__()
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.g_cond_mtd = g_cond_mtd
        self.mixed_precision = mixed_precision
        self.MODEL = MODEL
        self.in_dims = g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.num_blocks = len(self.in_dims)
        self.affine_input_dim = 0

        info_dim = 0
        if self.MODEL.info_type in ["discrete", "both"]:
            info_dim += self.MODEL.info_num_discrete_c*self.MODEL.info_dim_discrete_c
        if self.MODEL.info_type in ["continuous", "both"]:
            info_dim += self.MODEL.info_num_conti_c

        self.g_info_injection = self.MODEL.g_info_injection
        if self.MODEL.info_type != "N/A":
            if self.g_info_injection == "concat":
                self.info_mix_linear = MODULES.g_linear(in_features=self.z_dim + info_dim, out_features=self.z_dim, bias=True)
            elif self.g_info_injection == "cBN":
                self.affine_input_dim += self.z_dim
                self.info_proj_linear = MODULES.g_linear(in_features=info_dim, out_features=self.z_dim, bias=True)

        self.linear0 = MODULES.g_linear(in_features=self.z_dim, out_features=self.in_dims[0] * self.bottom * self.bottom, bias=True)

        if self.g_cond_mtd != "W/O" and self.g_cond_mtd == "cBN":
            self.affine_input_dim += self.num_classes

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index],
                         g_cond_mtd=self.g_cond_mtd,
                         g_info_injection=self.g_info_injection,
                         affine_input_dim=self.affine_input_dim,
                         MODULES=MODULES)
            ]]

            if index + 1 in attn_g_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=True, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = ops.batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = MODULES.g_act_fn
        self.conv2d5 = MODULES.g_conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        ops.init_weights(self.modules, g_init)

    def forward(self, z, label, shared_label=None, eval=False):
        affine_list = []
        if self.g_cond_mtd != "W/O":
            label = F.one_hot(label, num_classes=self.num_classes).to(torch.float32)
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            if self.MODEL.info_type != "N/A":
                if self.g_info_injection == "concat":
                    z = self.info_mix_linear(z)
                elif self.g_info_injection == "cBN":
                    z, z_info = z[:, :self.z_dim], z[:, self.z_dim:]
                    affine_list.append(self.info_proj_linear(z_info))

            if self.g_cond_mtd != "W/O":
                affine_list.append(label)
            if len(affine_list) > 0:
                affines = torch.cat(affine_list, 1)
            else:
                affines = None

            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, ops.SelfAttention):
                        act = block(act)
                    else:
                        act = block(act, affines)

            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out
