from gen_models import resnet, big_resnet, stylegan3
import torch
import gen_models.utils.sample as sample
import torch.nn.functional as F


def get_imagenet_models(device, img_size=128, num_classes=1000):
    """
    Load pretrained generative models of the ImageNet dataset. The pretrained weights are downloaded from
    the official repository of Studio GAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN and its hugging
    face page: https://huggingface.co/Mingguksky/PyTorch-StudioGAN.
    """
    magnitude_ema_beta = 0.5 ** (256 * 1 / (20 * 1e3))
    styleGAN3 = stylegan3.Generator(z_dim=512, c_dim=1000, w_dim=512, img_resolution=128, img_channels=3,
                                    info_num_discrete_c="N/A", info_dim_discrete_c="N/A", info_num_conti_c="N/A",
                                    info_type="N/A", mapping_kwargs={"num_layers": 2},
                                    synthesis_kwargs={"channel_base": 32768, "channel_max": 512,
                                                      "num_fp16_res": 0, "conv_clamp": None, "conv_kernel": 3,
                                                      "use_radial_filters": False,
                                                      "magnitude_ema_beta": magnitude_ema_beta}).to(device)
    # Download from https://huggingface.co/Mingguksky/PyTorch-StudioGAN/blob/main/studiogan_official_ckpt/ImageNet_tailored/ImageNet-StyleGAN3-t-SPD-train-2022_02_16_12_40_58/model%3DG_ema-best-weights-step%3D94000.pth
    ckpt_path_stylegan3 = './pretrained_weights/stylegan3=G_ema-best-weights-step=94000.pth'
    ckpt_stylegan3 = torch.load(ckpt_path_stylegan3, map_location=lambda storage, loc: storage)
    styleGAN3.load_state_dict(ckpt_stylegan3["state_dict"], strict=True)
    print('Successfully loading trained stylegan3!')

    SNGAN = resnet.Generator(z_dim=128, g_shared_dim='N/A', img_size=128, g_conv_dim=64, apply_attn=False,
                             attn_g_loc=['N/A'], g_cond_mtd='cBN', num_classes=1000,
                             g_init='ortho', g_depth='N/A', mixed_precision=False).to(device)

    # Download from https://huggingface.co/Mingguksky/PyTorch-StudioGAN/blob/main/studiogan_official_ckpt/ImageNet_tailored/ImageNet-SNGAN-256-train-2022_03_06_03_53_44/model%3DG-best-weights-step%3D472000.pth
    ckpt_path_SNGAN = './pretrained_weights/SNGAN=G-best-weights-step=472000.pth'
    ckpt_SNGAN = torch.load(ckpt_path_SNGAN, map_location=lambda storage, loc: storage)
    SNGAN.load_state_dict(ckpt_SNGAN["state_dict"], strict=True)
    print('Successfully loading trained SNGAN!')

    BigGAN = big_resnet.Generator(z_dim=120, g_shared_dim=128, img_size=img_size, g_conv_dim=96, apply_attn=True,
                                  attn_g_loc=[4], g_cond_mtd="cBN", num_classes=num_classes, g_init="ortho",
                                  g_depth="N/A",
                                  mixed_precision=False).to(device)
    # Download from https://drive.google.com/drive/folders/1_RTYZ0RXbVLWufE7bbWPvp8n_QJbA8K0?usp=sharing
    ckpt_path_BigGAN = './pretrained_weights/BigGAN=G_ema-best-weights-step=198000.pth'
    ckpt_BigGAN = torch.load(ckpt_path_BigGAN, map_location=lambda storage, loc: storage)
    BigGAN.load_state_dict(ckpt_BigGAN["state_dict"], strict=True)
    print('Successfully loading trained BigGAN!')

    ContraGAN = big_resnet.Generator(z_dim=120, g_shared_dim=128, img_size=img_size, g_conv_dim=96, apply_attn=True,
                                     attn_g_loc=[4], g_cond_mtd="cBN", num_classes=num_classes, g_init="ortho",
                                     g_depth="N/A", mixed_precision=False).to(device)
    # Download from https://drive.google.com/drive/folders/1pbP6LQ00VF7si-LXLvd_D00Pk5_E_JnP?usp=sharing
    ckpt_path_ContraGAN = './pretrained_weights/ContraGAN=G_ema-best-weights-step=198000.pth'
    ckpt_ContraGAN = torch.load(ckpt_path_ContraGAN, map_location=lambda storage, loc: storage)
    ContraGAN.load_state_dict(ckpt_ContraGAN["state_dict"], strict=True)
    print('Successfully loading trained ContraGAN!')

    # Download from https://drive.google.com/drive/folders/1lWw6Oh_Mjc7BKiSUKhWxfgP9QLc45g8a?usp=sharing
    ReACGAN = big_resnet.Generator(z_dim=120, g_shared_dim=128, img_size=img_size, g_conv_dim=96, apply_attn=True,
                                   attn_g_loc=[4], g_cond_mtd="cBN", num_classes=num_classes, g_init="ortho",
                                   g_depth="N/A", mixed_precision=False).to(device)
    ckpt_path_ReACGAN = './pretrained_weights/ReACGAN1=G_ema-best-weights-step=496000.pth'
    ckpt_ReACGAN = torch.load(ckpt_path_ReACGAN, map_location=lambda storage, loc: storage)
    ReACGAN.load_state_dict(ckpt_ReACGAN["state_dict"], strict=True)
    print('Successfully loading trained ReACGAN!')

    gen_list = [styleGAN3, SNGAN, BigGAN, ContraGAN, ReACGAN]

    def generate_fn(idx, bs, idc, gen_dim=128):
        if idc == 0:
            labels = torch.randint(low=0, high=num_classes, size=(bs,), dtype=torch.long, device=device)
            one_hot_fake_labels = F.one_hot(labels, num_classes=num_classes)
            gen_img = idx(
                z=sample.sample_normal(batch_size=bs, z_dim=512, truncation_factor=1, device=device),
                label=one_hot_fake_labels,
                eval=True
            )
        else:
            gen_img = idx(
                z=sample.sample_normal(batch_size=bs, z_dim=gen_dim, truncation_factor=1.0, device=device),
                label=torch.randint(low=0, high=num_classes, size=(bs,), dtype=torch.long, device=device),
                eval=True
            )
        return (gen_img + 1) / 2

    return gen_list, generate_fn

