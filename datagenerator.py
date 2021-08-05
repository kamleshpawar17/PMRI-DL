import os
import h5py
import numpy as np
import glob
from pygrappa import grappa
from skimage.measure import compare_ssim, compare_psnr
import matplotlib.pyplot as plt

def ssim(gt, pred):
    return compare_ssim(np.transpose(gt, (1, 2, 0)), np.transpose(pred, (1, 2, 0)), multichannel=True, data_range=gt.max())
                        
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
    print('Running predict_g4zf_all_slc .....')
    slc, r, c = img_g4.shape
    img_pred = np.zeros((slc, r, c))
    for k in range(slc):
        x0 = np.expand_dims(img_g4[k:k + 1, :, :], -1)
        x1 = np.expand_dims(img_ia[k:k + 1, :, :], -1)
        x = np.concatenate((x0, x1), axis=-1)
        img_pred[k, :, :] = np.squeeze(model.predict(x))
    return img_pred

def predict_all_slc(model, img_inp):
    print('Running predict_all_slc .....')
    slc, r, c = img_inp.shape
    img_pred = np.zeros((slc, r, c))
    for k in range(slc):
        x = np.expand_dims(img_inp[k:k + 1, :, :], -1)
        img_pred[k, :, :] = np.squeeze(model.predict(x))
    return img_pred

def generate_mask(num_cols, center_fractions, accelerations, seed=None):
    '''
    This function generates a 1d undersampling mask
    :param num_cols: Number of data points (phase encodes)
    :param center_fraction: fraction of the center k-space to be present
    :param acceleration: Acceleration factor
    :return: 1d sampling mask boolean
    '''
    if seed is not None:
        np.random.seed(seed)
    choice = np.random.randint(0, len(accelerations))
    center_fraction = center_fractions[choice]
    acceleration = accelerations[choice]

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = np.random.uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True
    return mask

def ifft2d(x, axes=(1, 2)):
    y = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)
    return y

def fft2d(x, axes=(1, 2)):
    y = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x, axes=axes), axes=axes, norm='ortho'), axes=axes)
    return y

def ifft1d(x, axes=-1):
    y = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axes), axis=axes, norm='ortho'), axes=axes)
    return y

def fft1d(x, axes=-1):
    y = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axes), axis=axes, norm='ortho'), axes=axes)
    return y

def center_crop(data, shape):
    if shape[0] <= data.shape[-2]:
        w_from = (data.shape[-2] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :]
    else:
        w_zpad = (shape[0] - data.shape[-2])/2.0
        w_zpadL, w_zpadR = int(np.floor(w_zpad)), int(np.ceil(w_zpad))
        if len(data.shape) == 3:
            data = np.pad(data, ((0, 0), (w_zpadL, w_zpadR), (0, 0)), mode='constant')
        else:
            data = np.pad(data, ((0, 0), (0, 0), (w_zpadL, w_zpadR), (0, 0)), mode='constant')

    if shape[1] <= data.shape[-1]:
        h_from = (data.shape[-1] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., h_from:h_to]
    else:
        h_zpad = (shape[1] - data.shape[-1])/2.0
        h_zpadL, h_zpadR = int(np.floor(h_zpad)), int(np.ceil(h_zpad))
        if len(data.shape) == 3:
            data = np.pad(data, ((0, 0), (0, 0), (h_zpadL, h_zpadR)), mode='constant')
        else:
            data = np.pad(data, ((0, 0), (0, 0), (0, 0), (h_zpadL, h_zpadR)), mode='constant')
    return data

def get_data_vdus(fname, seed=None, H=320, W=320, centreFrac=0.08, acc=4):
    f = h5py.File(fname, 'r')
    kspace = np.squeeze(np.array(f['kspace']))
    img_rss = np.squeeze(np.array(f['reconstruction_rss']))
    # rec_esc = np.array(f['reconstruction_esc'])
    C = kspace.shape[-1]
    # ---- Get 1D k-space ---- #
    kspace = ifft1d(kspace, axes=2)

    # ----- perform undersamling ------- #
    mask = generate_mask(C, [centreFrac], [acc], seed=seed)
    kspace_und = kspace * mask

    # ---- Reconstruct image from k-space ---- #
    img_und = np.abs(ifft1d(kspace_und, axes=3))
    img_und = center_crop(img_und, shape=(320, 320))
    img_ref = np.array(img_rss).copy()
    img_und = np.squeeze(np.sqrt(np.sum(img_und * img_und, 1)))
    sl, r, c = img_ref.shape

    # ---- Data Noramilization ---- #'
    normalize = normalize_std(img_und)

    img_und = center_crop(img_und, (H, W))
    img_ref = center_crop(img_ref, (H, W))
    img_rss = center_crop(img_rss, (H, W))
    img_und = normalize.normalize(img_und)
    img_ref = normalize.normalize(img_ref)
    return img_und, img_ref, img_rss, normalize

def grappaSimMutiSlice(fname, acc=4.0, acs=24, kernel_size=(5, 5), coil_axis=0):
    '''
    This function takes kspace data with mutiple slices and return the undersampled and grappa reconstructed kspace
    :param kspaceMs: Input kspace data
    :param acc: Acceleration factor
    :param acs: number of acs lines for grappa reconstruction
    :param kernel_size: kernel size for grappa recon
    :param coil_axis: axis of coil/channel dimension
    :return:
    '''
    print('Running Grappa simulation ....')
    file = h5py.File(fname, 'r')
    kspaceMs = np.array(file['kspace'], dtype=complex)
    slc, ch, fe, pe = kspaceMs.shape
    kspaceMsGrRec = np.zeros((slc, ch, fe, pe), dtype=complex)
    mask = np.zeros(pe, )
    mask[1:pe:int(acc)] = 1
    mask[int(pe / 2 - acs / 2):int(pe / 2 + acs / 2)] = 1
    calibMs = kspaceMs[:, :, :, int(pe / 2 - acs / 2):int(pe / 2 + acs / 2)].copy()
    kspaceMsUs = kspaceMs * mask
    for sl in range(slc):
        ks = kspaceMsUs[sl, :, :, :]
        calib = calibMs[sl, :, :, :]
        kspaceMsGrRec[sl, :, :, :] = grappa(ks, calib, kernel_size=kernel_size, coil_axis=coil_axis)
    return kspaceMsUs, kspaceMsGrRec, kspaceMs

def get_data_GrappaZfDL(fname, H=320, W=320, acc=4.0, acs=24, kernel_size=(5, 5)):
    kspaceMsUs, kspaceMsGrRec, kspaceMs = grappaSimMutiSlice(fname, acs=acs, acc=acc, kernel_size=kernel_size, coil_axis=0)
    Iref = ifft2d(kspaceMs, axes=(-1, -2))
    Ia = ifft2d(kspaceMsUs, axes=(-1, -2))
    Ig4 = ifft2d(kspaceMsGrRec, axes=(-1, -2))
    Iref = np.squeeze(np.sqrt(np.sum(np.abs(Iref) * np.abs(Iref), 1)))
    Iref = center_crop(Iref, (H, W))
    Ia = np.squeeze(np.sqrt(np.sum(np.abs(Ia) * np.abs(Ia), 1)))
    Ia = center_crop(Ia, (H, W))
    Ig4 = np.squeeze(np.sqrt(np.sum(np.abs(Ig4) * np.abs(Ig4), 1)))
    Ig4 = center_crop(Ig4, (H, W))
    # --- normalize --- #
    normalize = normalize_std(Ig4)
    Ig4 = normalize.normalize(Ig4)
    Ia = normalize.normalize(Ia)
    return Ig4, Ia, Iref, normalize

def displaysubplots(x_vd, y_pred_vd, y_vd, ssim_after_vd, psnr_after_vd, nmse_after_vd,
                    x_g4, y_g4, y_pred_g4, ssim_before_g4, psnr_before_g4, nmse_before_g4, ssim_after_g4, psnr_after_g4,
                    nmse_after_g4,
                    y_pred_g4zf, ssim_after_g4zf, psnr_after_g4zf, nmse_after_g4zf, slc, r, c, sz):
    img_zeros = np.zeros(x_vd[slc].shape)
    scale = 6.0

    # plt.figure(figsize=(13.85, 12))
    plt.figure(figsize=(14.8, 12))
    plt.rc('font', size=18)
    plt.subplot(351)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_g4[slc], cmap='gray')
    plt.xlabel('(SSIM, PSNR)')
    plt.title('Reference')

    plt.subplot(352)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_g4[slc], cmap='gray')
    temp = "(" + "{0:0.4f}".format(ssim_before_g4) + " , " + "{0:0.2f}".format(psnr_before_g4) + ")"
    plt.xlabel(temp)
    plt.title('Grappa')

    plt.subplot(353)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred_vd[slc], cmap='gray')
    temp = "(" + "{0:0.4f}".format(ssim_after_vd) + " , " + "{0:0.2f}".format(psnr_after_vd) + ")"
    plt.xlabel(temp)
    plt.title('Vdus')

    plt.subplot(354)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred_g4[slc], cmap='gray')
    temp = "(" + "{0:0.4f}".format(ssim_after_g4) + " , " + "{0:0.2f}".format(psnr_after_g4) + ")"
    plt.xlabel(temp)
    plt.title('Grappa DL')

    plt.subplot(355)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred_g4zf[slc], cmap='gray')
    temp = "(" + "{0:0.4f}".format(ssim_after_g4zf) + " , " + "{0:0.2f}".format(psnr_after_g4zf) + ")"
    plt.xlabel(temp)
    plt.title('GrappaZf DL')

    plt.subplot(356)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_zeros, cmap='gray')
    plt.xlabel('NMSE')
    # plt.title('Difference (6x)')

    plt.subplot(357)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(scale * np.abs(x_g4[slc] - y_g4[slc]), cmap='gray', vmin=0, vmax=np.max(y_g4))
    temp = "{0:0.4f}".format(nmse_before_g4)
    plt.xlabel(temp)
    # plt.title('difference (6x)')

    plt.subplot(358)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(scale * np.abs(y_pred_vd[slc] - y_vd[slc]), cmap='gray', vmin=0, vmax=np.max(y_g4))
    temp = "{0:0.4f}".format(nmse_after_vd)
    plt.xlabel(temp)
    # plt.title('difference (6x)')

    plt.subplot(359)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(scale * np.abs(y_pred_g4[slc] - y_g4[slc]), cmap='gray', vmin=0, vmax=np.max(y_g4))
    temp = "{0:0.4f}".format(nmse_after_g4)
    plt.xlabel(temp)
    # plt.title('difference (6x)')

    plt.subplot(3, 5, 10)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(scale * np.abs(y_pred_g4zf[slc] - y_g4[slc]), cmap='gray', vmin=0, vmax=np.max(y_g4))
    temp = "{0:0.4f}".format(nmse_after_g4zf)
    plt.xlabel(temp)
    # plt.title('difference (6x)')

    plt.subplot(3, 5, 11)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_g4[slc, r:r + sz, c:c + sz], cmap='gray')
    # plt.title('enlarged ROI')

    plt.subplot(3, 5, 12)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_g4[slc, r:r + sz, c:c + sz], cmap='gray')
    # plt.title('enlarged ROI')

    plt.subplot(3, 5, 13)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred_vd[slc, r:r + sz, c:c + sz], cmap='gray')
    # plt.title('enlarged ROI')

    plt.subplot(3, 5, 14)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred_g4[slc, r:r + sz, c:c + sz], cmap='gray')
    # plt.title('enlarged ROI')

    plt.subplot(3, 5, 15)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_pred_g4zf[slc, r:r + sz, c:c + sz], cmap='gray')
    # plt.title('enlarged ROI')
    plt.subplots_adjust(left=0.0025, bottom=0.005, right=0.9975, top=0.965, wspace=0.0, hspace=0.12)





