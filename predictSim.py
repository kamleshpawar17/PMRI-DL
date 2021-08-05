import sys
import numpy as np
from encoder_decoder_grappadl import Unet_enc_dec_v1
import matplotlib.pyplot as plt
from datagenerator import ssim, psnr, nmse, get_data_vdus, get_data_GrappaZfDL, predict_all_slc, predict_g4zf_all_slc, displaysubplots

if __name__ == '__main__':
    # --- data paths --- #
    assert len(sys.argv) == 3, "wrong number of arguments"
    antomy = str(sys.argv[1])
    assert antomy in ['knee', 'brain'], "Anatomy must be 'knee' or 'brain'"
    fname = str(sys.argv[2])
    
    # --- load model ---- #
    model_g4zf = Unet_enc_dec_v1(n_channel=64, nFeatIn=2, nFeatOut=1, H=320, W=320, k_size=3, isRes=False)
    if antomy=='knee':
        model_g4zf.load_weights('./weights/knee/model-g4zf.hdf5')
    else:
        model_g4zf.load_weights('./weights/brain/model-g4zf.hdf5')

    model_g4 = Unet_enc_dec_v1(n_channel=64, nFeatIn=1, nFeatOut=1, H=320, W=320, k_size=3, isRes=False)
    if antomy=='knee':
        model_g4.load_weights('./weights/knee/model-g4.hdf5')
    else:
        model_g4.load_weights('./weights/brain/model-g4.hdf5')

    model_vd = Unet_enc_dec_v1(n_channel=64, nFeatIn=1, nFeatOut=1, H=320, W=320, k_size=3, isRes=False)
    if antomy=='knee':
        model_vd.load_weights('./weights/knee/model-vdus.hdf5')
    else:
        model_vd.load_weights('./weights/brain/model-vdus.hdf5')

    # --- load data --- #
    img_vd, _, img_refvd, normalizevd = get_data_vdus(fname, seed=113, H=320, W=320, centreFrac=0.08, acc=4.0)
    img_g4, img_ia, img_ref, normalize = get_data_GrappaZfDL(fname)

    # --- pred g4zf --- #
    img_pred_g4zf = normalize.denormalize(predict_g4zf_all_slc(model_g4zf, img_g4, img_ia)) 

    # --- pred g4 --- #
    img_pred_g4 = normalize.denormalize(predict_all_slc(model_g4, img_g4)) 
    
    # --- pred vdus --- #
    img_pred_vd = normalizevd.denormalize(predict_all_slc(model_vd, img_vd))

    # ---- compute scores --- #
    img_g4 = normalize.denormalize(img_g4)
    ssim_before_g4 = ssim(img_ref, img_g4)
    psnr_before_g4 = psnr(img_ref, img_g4)
    nmse_before_g4 = nmse(img_ref, img_g4)
    ssim_after_g4 = ssim(img_ref, img_pred_g4)
    psnr_after_g4 = psnr(img_ref, img_pred_g4)
    nmse_after_g4 = nmse(img_ref, img_pred_g4)
    ssim_after_g4zf = ssim(img_ref, img_pred_g4zf)
    psnr_after_g4zf = psnr(img_ref, img_pred_g4zf)
    nmse_after_g4zf = nmse(img_ref, img_pred_g4zf)
    ssim_after_vd = ssim(img_refvd, img_pred_vd)
    psnr_after_vd = psnr(img_refvd, img_pred_vd)
    nmse_after_vd = nmse(img_refvd, img_pred_vd)

    sz = 150
    slc, r, c = img_g4.shape[0]//2, img_g4.shape[1]//2-sz//2, img_g4.shape[2]//2-sz//2
    # --- plot --- #
    displaysubplots(img_vd, img_pred_vd, img_refvd, ssim_after_vd, psnr_after_vd, nmse_after_vd, img_g4, img_ref, img_pred_g4, ssim_before_g4, psnr_before_g4, nmse_before_g4, ssim_after_g4, psnr_after_g4, nmse_after_g4, img_pred_g4zf, ssim_after_g4zf, psnr_after_g4zf, nmse_after_g4zf, slc, r, c, sz)
    plt.show()






  
