src_image=""
ref_img=""
aa={"input_dir": "media/input", "output_dir": "media/output", "im_path1": src_image,
                         "im_path2": src_image,
                         "im_path3": ref_img,
                         "sign": "realistic", "smooth": 5, "size": 1024, "ckpt": "/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/celebrity_model-master/Barbershop/pretrained_models/ffhq.pt",
                         "channel_multiplier": 2, "latent": 512, "n_mlp": 8,
                         "device": "cuda", "seed": None, "tile_latent": None, "opt_name": "adam", "learning_rate": 0.01,
                         "lr_schedule": "fixed",
                         "save_intermediate": None, "save_interval": 300, "verbose": None, "seg_ckpt": "/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/celebrity_model-master/Barbershop/pretrained_models/seg.pth",
                         "percept_lambda": 1.0,
                         "l2_lambda": 1.0, "p_norm_lambda": 0.001, "l_F_lambda": 0.1, "W_steps": 1100, "FS_steps": 250,
                         "ce_lambda": 1.0, "style_lambda": 4e4,
                         "align_steps1": 140, "align_steps2": 100, "face_lambdat": 1.0, "hair_lambda": 1.0, "blend_steps": 40}

print(type(aa))