plot_names = {
    'r50_ckpt=None': 'ResNet-50',
    'r18_ckpt=None': 'ResNet-18',
    'vit_s_ckpt=None': 'ViT-S',
    'vit_b_ckpt=None': 'ViT-B',
    'vit_l_ckpt=None': 'ViT-L',
    'nb_din_b_s483_ckpt=last': 'DINO',
    # 'nb_din_b_s483_ckpt=last': 'DINO_s483',
    'nb_din_b_s873_ckpt=last': 'DINO_s873',
    'nb_mae_b_s483_ckpt=last': 'MAE',
    # 'nb_mae_b_s483_ckpt=last': 'MAE_s483',
    'nb_mae_b_s873_ckpt=last': 'MAE_s873',
    'nb_exc_wb_v0_r18_pt_s483_rh_ckpt=last': 'NAB-WB_R18_PT',
    'nb_exc_wb_v0_r18_fs_s483_rh_ckpt=last': 'NAB-WB_R18_FS',
    'nb_exc_wb2_v0_r18_fs_s483_rh_ckpt=last': 'R18 NAB-WB',
    'nb_pt_r18_s483_ckpt=last': 'R18 ImgNet PT',
    'nb_fs_r18_s483_ckpt=last': 'R18 s483',
    'nb_r18_fs_s483_rh_nbsc_ckpt=last': 'R18 s483',
    'nbsc_r18_fs_s483_rh_nbsc_ckpt=last': 'R18 NAB+SC',
    'nb_fs_r18_s873_ckpt=last': 'R18 s873',
}


def shrink_cbar(ax, shrink=0.9):
    b = ax.get_position()
    new_h = b.height*shrink
    pad = (b.height-new_h)/2.
    new_y0 = b.y0 + pad
    new_y1 = b.y1 - pad
    b.y0 = new_y0
    b.y1 = new_y1
    ax.set_position(b)