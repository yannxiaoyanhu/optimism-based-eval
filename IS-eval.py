import torch
from scipy.stats import entropy
from termcolor import colored
import numpy as np
import torchvision
import torchvision.transforms.functional as F
import torch.nn.functional as nnf
from torch import Tensor
from gen_models import get_models


# Configs
dataset = 'imagenet'
strategy = str(input("Eval strategy (UCB/naive-UCB/greedy): "))
T_step = 1000
bs = 5
C_prime = 1.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inception feat_model
inception_model = torchvision.models.inception_v3(num_classes=1000, init_weights=False).to(device)
inception_model.load_state_dict(torch.load('./feat_models/pretrained_weights/inception_v3_google-0cc3c7bd.pth'))
num_cls = 1000

# Prepare generative models and statistics
N = 5
Gen_list, generate_fn = get_models.get_imagenet_models(device=device)
IS = np.load('./stats/imagenet_IS_Eval.npy')
optimal_index = np.argmax(IS)


if __name__ == '__main__':
    np.random.seed(1234)
    num_epoch = 20
    failure_prop = 0.05
    reg_list = np.zeros(T_step + 1)
    mean_pull = np.zeros((T_step, N))

    for epoch in range(1, num_epoch + 1):
        hat_IS = np.zeros(N) + np.inf
        empirical_IS = np.zeros(N) + np.inf
        pull_count = np.zeros(N)                                    # visitation
        regret = 0.
        pred_dist = np.zeros((N, 1000))                             # marginal class dist.
        pred_dist_list = np.empty((N, int(T_step * bs), 1000))
        cond_ent = np.zeros(N)                                      # H(Y_g|X_g)
        cond_ent_list = [[] for _ in range(N)]

        # Start of online steps
        for t in range(1, T_step + 1):

            # Pick generator g_t according the strategy
            if strategy in ['UCB', 'nUCB']:
                select_model = np.argmax(hat_IS)
            elif strategy == 'greedy':
                select_model = np.argmax(empirical_IS)
            else:
                raise NotImplementedError

            # Query a batch of generated images from g_t and extract the features
            with torch.no_grad():
                gen_img = generate_fn(idx=Gen_list[select_model], bs=bs, idc=select_model)
                trans_img = F.resize((gen_img * 255).to(torch.uint8), 342)
                trans_img = F.center_crop(trans_img, 299)
                if not isinstance(trans_img, Tensor):
                    trans_img = F.pil_to_tensor(trans_img)
                trans_img = F.convert_image_dtype(trans_img, torch.float)
                pred = nnf.softmax(inception_model(trans_img)[0], dim=1).detach().cpu().numpy()

            # Update statistics
            regret += np.max(IS) - IS[select_model]                 # instant regret
            reg_list[t] += regret                                   # average over epochs
            pull_count[select_model] += 1
            pull_prop = pull_count / t
            mean_pull[t - 1] += pull_prop
            nt = pull_count[select_model]
            n_sample = int(nt * bs)
            pred_dist_list[select_model][n_sample - bs:n_sample] = pred
            prev_accumulative_cond_ent = cond_ent[select_model] * (n_sample - bs)
            for i in range(bs):
                ent_i = entropy(pred[i])
                prev_accumulative_cond_ent += ent_i   # conditional entropy
                cond_ent_list[select_model].append(ent_i)
            cond_ent[select_model] = prev_accumulative_cond_ent / n_sample
            pred_dist[select_model] = ((n_sample - bs) * pred_dist[select_model] +
                                       np.sum(pred, axis=0, keepdims=True)) / n_sample  # marginal class dist.

            # Empirical IS
            empirical_IS[select_model] = entropy(pred_dist[select_model]) - cond_ent[select_model]

            if strategy == 'UCB':       # IS-UCB
                sig_cls = np.var(pred_dist_list[select_model][:n_sample], axis=0)
                e0 = np.sqrt(sig_cls / n_sample * np.log(num_cls / failure_prop)) + \
                     np.log(num_cls / failure_prop) / (n_sample - 1)
                sig2_cond_ent = np.var(np.array(cond_ent_list[select_model]))
                e1 = np.sqrt(sig2_cond_ent / n_sample * np.log(1 / failure_prop)) + \
                    C_prime * np.log(1 / failure_prop) / (n_sample - 1)         # conditional ent.

                opt_dist = pred_dist[select_model]      # Compute the optimistic marginal distribution
                for j in range(1000):
                    delta_j = ((1 / np.e) - opt_dist[j]) / np.abs((1 / np.e) - opt_dist[j]) * e0[j]
                    if delta_j > np.abs((1 / np.e) - opt_dist[j]):
                        opt_dist[j] = 1 / np.e
                    else:
                        opt_dist[j] += delta_j
                full_ent = - opt_dist.dot(np.log(opt_dist))     # optimistic marginal entropy

                hat_IS[select_model] = full_ent - (cond_ent[select_model] - e1)

            elif strategy == 'naive-UCB':        # naive UCB
                e0 = np.sqrt(1 / (2 * n_sample) * np.log(4 * num_cls / failure_prop)) * np.ones(1000)
                e1 = C_prime * np.sqrt(1 / (2 * n_sample) * np.log(4 / failure_prop))

                # Compute optimistic unconditional entropy H(Y_g)
                opt_dist = pred_dist[select_model]
                for j in range(1000):
                    delta_j = ((1 / np.e) - opt_dist[j]) / np.abs((1 / np.e) - opt_dist[j]) * e0[j]
                    if delta_j > np.abs((1 / np.e) - opt_dist[j]):
                        opt_dist[j] = 1 / np.e
                    else:
                        opt_dist[j] += delta_j
                full_ent = - opt_dist.dot(np.log(opt_dist))

                hat_IS[select_model] = full_ent - (cond_ent[select_model] - e1)

            if t == 1 or t % 100 == 0:
                print(colored('dataset: {}, metric: IS, strategy: {}, epoch: {}, step {}, '
                              'average regret: {}, average optimal pick-ratio: {}, average pick-ratio: {}, '
                              .format(
                               dataset, strategy, epoch, t,
                               reg_list[t] / epoch, mean_pull[t - 1][optimal_index] / epoch, mean_pull[t - 1] / epoch),
                      'red'), '\n')

        np.savez('./results/ISEval_{}_{}.npz'.format(dataset, strategy),
                 mean_regret=reg_list / epoch, mean_pull=mean_pull / epoch)
