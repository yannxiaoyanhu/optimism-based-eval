import numpy as np
from scipy import linalg
import torch
from termcolor import colored
from torch.nn.functional import adaptive_avg_pool2d
import open_clip

from gen_models import get_models
from feat_models.vision_transformer import vit_large, get_dinov2_tform
from feat_models.transform import image_transform_v2
from feat_models.inception import InceptionV3
from aux import compute_FID

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

# Configs
dataset = 'imagenet'
strategy = 'UCB'                                                # UCB/naive-UCB/greedy
embed_model = 'inception-v3'                                    # inception-v3/dinov2/clip
ths_ratio = 0.05
T_step = 1000                                                   # evaluation step
bs = 5                                                          # batch size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare the embedding pretrained model (download and save under the directory ./feat_models/pretrained weights)
if embed_model == 'inception-v3':
    f_dim = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[f_dim]
    inception_model = InceptionV3([block_idx]).to(device)
elif embed_model == 'dinov2':
    f_dim = 1024
    dinov2_model = vit_large(patch_size=14, init_values=1.0, img_size=526, block_chunks=0).to(device)
    dinov2_model.load_state_dict(torch.load('./feat_models/pretrained_weights/dinov2_vitl14_pretrain.pth'))
    tform = get_dinov2_tform()
elif embed_model == 'clip':
    f_dim = 512
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
    clip_model.to(device)
    clip_model.load_state_dict(torch.load('./feat_models/pretrained_weights/vit_b_32-laion2b_e16-af8dbd0c.pth'))
    clip_model.eval()
    tform = image_transform_v2()
else:
    raise NotImplementedError

# Prepare statistics of real images and generators
N = 5
Gen_list, generate_fn = get_models.get_imagenet_models(device=device)

stat_d = np.load('./stats/imagenet_fid_stat_{}.npz'.format(embed_model))
fid = np.load('./stats/imagenet_FID_Eval_{}.npy'.format(embed_model))
optimal_index = np.argmin(fid)

# FID statistics of the real images
mu_r, sig_r = stat_d['mu'][:], stat_d['sigma'][:]
w_data = linalg.eigh(sig_r, eigvals_only=True)
norm2_sig_r = w_data[-1]
ths = ths_ratio * norm2_sig_r                                   # truncation parameter \tau

truncate_w_data = w_data
truncate_w_data[truncate_w_data < ths] = 0
tr_root_sig_r = np.sum(np.sqrt(truncate_w_data))


if __name__ == '__main__':
    np.random.seed(1234)
    num_epoch = 20                                              # results are averaged over 20 trials
    failure_prop = 0.05
    reg_list = np.zeros(T_step + 1)
    mean_pull = np.zeros((T_step, N))

    for epoch in range(1, num_epoch + 1):
        # Initialization
        hat_fid = np.zeros(N) - np.inf                          # estimated FID, initialized to be -inf
        tilde_fid = np.zeros(N) - np.inf                        # empirical FID, initialized to be -inf
        pull_count = np.zeros(N)                                # visitation
        gen_img_embs = [np.empty((1, 1)) for _ in range(N)]     # embedding vectors of the generated samples
        regret = 0.

        # Start a trial of online evaluation
        for t in range(1, T_step + 1):

            # Pick generator g_t according the strategy
            if strategy in ['UCB', 'nUCB']:
                select_model = np.argmin(hat_fid)
            elif strategy == 'greedy':
                select_model = np.argmin(tilde_fid)             # greedy policy
            else:
                raise NotImplementedError

            # Query a batch of generated images from g_t and extract the embeddings
            gen_img = generate_fn(idx=Gen_list[select_model], bs=bs, idc=select_model)
            img_tform = gen_img if embed_model == 'inception-v3' else tform(gen_img)
            with torch.no_grad():
                if embed_model == 'inception-v3':
                    gen_img_feat = inception_model(img_tform)[0]
                    if gen_img_feat.size(2) != 1 or gen_img_feat.size(3) != 1:
                        gen_img_feat = adaptive_avg_pool2d(gen_img_feat, output_size=(1, 1))
                    gen_img_feat = gen_img_feat.cpu().detach().numpy().squeeze(3).squeeze(2)
                elif embed_model == 'dinov2':
                    gen_img_feat = dinov2_model(img_tform).cpu().detach().numpy()
                elif embed_model == 'clip':
                    gen_img_feat = clip_model.encode_image(img_tform).cpu().numpy()
                else:
                    raise NotImplementedError

            # Update statistics
            if pull_count[select_model] == 0:
                gen_img_embs[select_model] = gen_img_feat
            else:
                gen_img_embs[select_model] = np.concatenate((gen_img_embs[select_model], gen_img_feat), axis=0)
            regret += fid[select_model] - np.min(fid)       # instant regret
            reg_list[t] += regret
            pull_count[select_model] += 1                   # visitation
            pull_prop = pull_count / t                      # pick-ratio
            mean_pull[t - 1] += pull_prop
            nt = gen_img_embs[select_model].shape[0]          # number of generated samples

            # Compute empirical FID
            mu_gt = np.mean(gen_img_embs[select_model], axis=0)
            sigma_gt = np.cov(gen_img_embs[select_model], rowvar=False)
            tilde_fid[select_model] = compute_FID(mu1=mu_gt, mu2=mu_r, sigma1=sigma_gt, sigma2=sig_r)

            if strategy == 'UCB':                           # Implementation of FID-UCB

                L = np.sqrt(np.log(1 / failure_prop))
                # concentration of \mu_g
                sig2_max = np.max(np.diagonal(sigma_gt)) + L * (1 / (2 * nt)) ** 0.5
                norm2_sig_gt = sig2_max                     # estimate 2-norm by the largest per-entry variance
                truncate_sig = sigma_gt                     # truncate the empirical sigma_gt
                truncate_sig[np.abs(sigma_gt) < ths] = 0
                tr_sig_gt = np.trace(np.matmul(truncate_sig, sig_r))
                I_gt = np.sum(np.abs(truncate_sig))
                Bg1 = min(np.sqrt(2 * I_gt * L / nt), L * np.sqrt(sig2_max * f_dim / (2 * nt)))

                # concentration of \Sigma_g
                r_sig = tr_sig_gt / norm2_sig_gt
                Bg2 = 20 * sig2_max * norm2_sig_gt * np.sqrt((4 * r_sig + L ** 2) / nt) + Bg1 ** 2

                c = np.mean(np.sqrt(np.sum((gen_img_embs[select_model] - mu_gt) ** 2, axis=1, keepdims=True)))
                C_g = 2 * (0.5 + Bg1 + c)
                term1 = C_g * Bg1
                term2 = tr_sig_gt * np.sqrt(8 / nt) * L
                term3 = tr_root_sig_r * np.sqrt(Bg2)
                bonus = term1 + term2 + term3

                hat_fid[select_model] = tilde_fid[select_model] - bonus

            elif strategy == 'naive-UCB':                        # Implementation of naive-UCB

                # Estimate parameters in the bonus
                sig2_max = np.max(np.diagonal(sigma_gt))
                norm2_sig_gt = sig2_max
                tr_sig_gt = np.trace(sigma_gt)

                # concentration of \mu_g
                Bg1 = np.sqrt(sig2_max * f_dim / (2 * nt) * np.log(f_dim / failure_prop))
                # concentration of \Sigma_g
                q = (f_dim / nt) ** 0.5 + (1 / (2 * nt) * np.log(6. / failure_prop)) ** 0.5
                Bg2 = norm2_sig_gt * (2 * q + q ** 2) + Bg1 ** 2

                c = np.mean(np.sqrt(np.sum((gen_img_embs[select_model] - mu_gt) ** 2, axis=1, keepdims=True)))
                diff = mu_gt - mu_r
                C_g = 2 * (np.sqrt(diff.dot(diff)) + 2 * Bg1 + c)
                term1 = C_g * Bg1
                term2 = tr_sig_gt * np.sqrt(8 / nt * np.log(6 * f_dim / failure_prop))
                term3 = tr_root_sig_r * np.sqrt(8 * Bg2)
                bonus = term1 + term2 + term3

                hat_fid[select_model] = tilde_fid[select_model] - bonus

            # Print results
            if t == 1 or t % 20 == 0:
                print(colored('dataset: {}, metric: FID-{}, eval alg: {}, ths-ratio: {}, epoch: {}, '
                              'step {}, average regret: {}, average reg/step: {}, average optimal-pick ratio: {}, '
                              'average pick-ratio: {}'.format(
                               dataset, embed_model, strategy, ths_ratio if strategy == 'UCB' else 'NA', epoch,
                               t, reg_list[t] / epoch, reg_list[t] / (t * epoch),
                               mean_pull[t - 1][optimal_index] / epoch,
                               mean_pull[t - 1] / epoch), 'red'), '\n')

        np.savez('./results/FIDEval_{}_{}_{}.npz'.format(dataset, strategy, embed_model),
                 mean_regret=reg_list / epoch, mean_pull=mean_pull / epoch)
