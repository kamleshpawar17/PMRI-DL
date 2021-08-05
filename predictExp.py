import numpy as np
import matplotlib.pyplot as plt

from encoder_decoder_grappadl import Unet_enc_dec_v1
from datagenerator import normalize_std, ssim, nmse, psnr, predict_g4zf_all_slc, predict_all_slc, saturate_to_val
from scipy import ndimage


if __name__ == '__main__':
    # --- load model ---- #
    model_g4zf = Unet_enc_dec_v1(n_channel=64, nFeatIn=2, nFeatOut=1, H=320, W=320, k_size=3, isRes=False)
    model_g4zf.load_weights('./weights/knee/model-g4zf.hdf5')

    model_g4 = Unet_enc_dec_v1(n_channel=64, nFeatIn=1, nFeatOut=1, H=320, W=320, k_size=3, isRes=False)
    model_g4.load_weights('./weights/knee/model-g4.hdf5')

    # ----- load data ---- 3
    f = np.load('./data/exp_grappa4/ReconExpImg.npz')
    img_ref, img_g4, img_ia = f['IFs'], f['Ig4'], f['Ia']
    normalize = normalize_std(img_ref)
    img_g4 = normalize.normalize(img_g4)
    img_ia = normalize.normalize(img_ia)
    img_ref = normalize.normalize(img_ref)

    # --- Saturate --- #
    sat_val = np.minimum(np.max(img_ref), np.max(img_g4))
    img_g4 = saturate_to_val(img_g4, sat_val)
    img_ref = saturate_to_val(img_ref, sat_val)
    img_ia = saturate_to_val(img_ia, sat_val)

    # ---- Predict ---- #
    img_pred_g4zf = predict_g4zf_all_slc(model_g4zf, img_g4, img_ia)
    img_pred_g4 = predict_all_slc(model_g4, img_g4)

    # ---- imovement correction between two seperate scans --- #
    img_pred_g4zf = ndimage.shift(img_pred_g4zf, [0.0, -0.5, 0.5], mode='wrap')
    img_pred_g4 = ndimage.shift(img_pred_g4, [0.0, -0.5, 0.5], mode='wrap')
    img_g4 = ndimage.shift(img_g4, [0.0, -0.5, 0.5], mode='wrap')

    # ---- compute scores --- #
    ssim_before_g4 = ssim(img_ref, img_g4)
    psnr_before_g4 = psnr(img_ref, img_g4)
    nmse_before_g4 = nmse(img_ref, img_g4)
    ssim_after_g4 = ssim(img_ref, img_pred_g4)
    psnr_after_g4 = psnr(img_ref, img_pred_g4)
    nmse_after_g4 = nmse(img_ref, img_pred_g4)
    ssim_after_g4zf = ssim(img_ref, img_pred_g4zf)
    psnr_after_g4zf = psnr(img_ref, img_pred_g4zf)
    nmse_after_g4zf = nmse(img_ref, img_pred_g4zf)

    print('Grappa: ', ssim_before_g4, ssim_after_g4, psnr_before_g4, psnr_after_g4, nmse_before_g4, nmse_after_g4)
    print('Grappa: ', ssim_before_g4, ssim_after_g4zf, psnr_before_g4, psnr_after_g4zf, nmse_before_g4, nmse_after_g4zf)

    # --- extract a slice to display Grappa DL ---- #
    sl, r, c, sz = 15, 120, 100, 100
    for sl in range(sl, sl+1):
        y_g4 = np.squeeze(img_ref[sl, :, :])
        x_g4 = np.squeeze(img_g4[sl, :, :])
        y_pred_g4 = np.squeeze(img_pred_g4[sl, :, :])
        y_pred_g4zf = np.squeeze(img_pred_g4zf[sl, :, :])

        plt.figure(figsize=(11.8, 12))
        plt.rc('font', size=18)
        plt.subplot(341)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_g4, cmap='gray')
        plt.xlabel('(SSIM, PSNR)')
        plt.title('Reference')

        plt.subplot(342)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_g4, cmap='gray')
        temp = "(" + "{0:0.4f}".format(ssim_before_g4) + " , " + "{0:0.2f}".format(psnr_before_g4) + ")"
        plt.xlabel(temp)
        plt.title('Grappa')

        plt.subplot(343)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_pred_g4, cmap='gray')
        temp = "(" + "{0:0.4f}".format(ssim_after_g4) + " , " + "{0:0.2f}".format(psnr_after_g4) + ")"
        plt.xlabel(temp)
        plt.title('Grappa DL')

        plt.subplot(344)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_pred_g4zf, cmap='gray')
        temp = "(" + "{0:0.4f}".format(ssim_after_g4zf) + " , " + "{0:0.2f}".format(psnr_after_g4zf) + ")"
        plt.xlabel(temp)
        plt.title('GrappaZf DL')

        img_zeros = np.zeros(x_g4.shape)
        plt.subplot(345)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_zeros, cmap='gray')
        plt.xlabel('NMSE')
        # plt.title('Ref Difference (6x)')

        plt.subplot(346)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(6 * np.abs(x_g4 - y_g4), cmap='gray', vmin=0, vmax=np.max(y_g4))
        temp = "{0:0.4f}".format(nmse_before_g4)
        plt.xlabel(temp)
        # plt.title('Grappa Difference (6x)')

        plt.subplot(347)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(6 * np.abs(y_pred_g4 - y_g4), cmap='gray', vmin=0, vmax=np.max(y_g4))
        temp = "{0:0.4f}".format(nmse_after_g4)
        plt.xlabel(temp)
        # plt.title('Grappa DL Difference (6x)')

        plt.subplot(348)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(6 * np.abs(y_pred_g4zf - y_g4), cmap='gray', vmin=0, vmax=np.max(y_g4))
        temp = "{0:0.4f}".format(nmse_after_g4zf)
        plt.xlabel(temp)
        # plt.title('Difference (6x)')

        plt.subplot(3, 4, 9)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_g4[r:r + sz, c:c + sz], cmap='gray')
        # plt.title('Enlarged ROI')

        plt.subplot(3, 4, 10)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_g4[r:r + sz, c:c + sz], cmap='gray')
        # plt.title('Enlarged ROI')

        plt.subplot(3, 4, 11)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_pred_g4[r:r + sz, c:c + sz], cmap='gray')
        # plt.title('Enlarged ROI')

        plt.subplot(3, 4, 12)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_pred_g4zf[r:r + sz, c:c + sz], cmap='gray')
        # plt.title('Enlarged ROI')
        plt.subplots_adjust(left=0.0025, bottom=0.005, right=0.9975, top=0.965, wspace=0.0, hspace=0.12)
        plt.show()
