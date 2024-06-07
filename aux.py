import numpy as np
from scipy import linalg
import torch
from PIL import Image


def compute_FID(mu1, mu2, sigma1, sigma2, eps=1e-6):
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(np.matmul(sigma1, sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fd calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        """if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))"""
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    # Return mean and covariance terms and intermediate steps, as well as FD
    mean_term = diff.dot(diff)
    tr1, tr2 = np.trace(sigma1), np.trace(sigma2)
    cov_term = tr1 + tr2 - 2 * tr_covmean

    return mean_term + cov_term


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "JPEG", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img



