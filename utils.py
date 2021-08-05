import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr

def ssim(gt, pred):
    return compare_ssim(gt, pred, multichannel=True, data_range=gt.max())
                        
def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt, pred):
    return compare_psnr(gt, pred, data_range=gt.max())

class normalize_std():
    def __init__(self, x):
        self.eps = 1e-12
        self.sigma_src = np.std(x)

    def normalize(self, x):
        y = x / (self.eps + self.sigma_src)
        return y

    def denormalize(self, x):
        y = self.sigma_src * x
        return y

def saturate_to_val(x, val):
    x[x > val] = val
    return x

def predict_g4zf_all_slc(model, img_g4, img_ia):
    print('Running GrappaZfDL .....')
    slc, r, c = img_g4.shape
    img_pred = np.zeros((slc, r, c))
    for k in range(slc):
        x0 = np.expand_dims(img_g4[k:k + 1, :, :], -1)
        x1 = np.expand_dims(img_ia[k:k + 1, :, :], -1)
        x = np.concatenate((x0, x1), axis=-1)
        img_pred[k, :, :] = np.squeeze(model.predict(x))
    return img_pred

def predict_all_slc(model, img_inp):
    print('Running GrappaDL .....')
    slc, r, c = img_inp.shape
    img_pred = np.zeros((slc, r, c))
    for k in range(slc):
        x = np.expand_dims(img_inp[k:k + 1, :, :], -1)
        img_pred[k, :, :] = np.squeeze(model.predict(x))
    return img_pred
