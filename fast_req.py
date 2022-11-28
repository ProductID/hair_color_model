import requests

# new_d = {'input_dir': 'Barbershop/input/face', 'output_dir': 'Barbershop/output', 'im_path1': '18.png',
#          'im_path2': '18.png', 'im_path3': '11.png', 'sign': 'realistic', 'smooth': 5, 'size': 1024,
#          'ckpt': 'Barbershop/pretrained_models/ffhq.pt', 'channel_multiplier': 2, 'latent': 512, 'n_mlp': 8,
#          'device': 'cuda', 'seed': None, 'tile_latent': None, 'opt_name': 'adam', 'learning_rate': 0.01,
#          'lr_schedule': 'fixed', 'save_intermediate': None, 'save_interval': 300, 'verbose': None,
#          'seg_ckpt': 'Barbershop/pretrained_models/seg.pth', 'percept_lambda': 1.0, 'l2_lambda': 1.0,
#          'p_norm_lambda': 0.001, 'l_F_lambda': 0.1, 'W_steps': 1100, 'FS_steps': 250, 'ce_lambda': 1.0,
#          'style_lambda': 40000.0, 'align_steps1': 140, 'align_steps2': 100, 'face_lambdat': 1.0, 'hair_lambda': 1.0,
#          'blend_steps': 40}
# images.jpeg
# images1.jpeg
# index.jpeg
input_image='18.png'
ref_image='11.png'
# input_image='https://github.com/manjinderwebtunix/images_data/blob/main/14.png'
# ref_image='https://github.com/manjinderwebtunix/images_data/blob/main/blonde_ref.png'


new_d = {'im_path1': input_image,'im_path2': input_image, 'im_path3': ref_image }
# response = requests.post('http://3.120.107.31/hair_color_api', data=new_d,verify=False)
# response = requests.post('http://127.0.0.1:8000/hair_color_api', data=new_d, verify=False)
response = requests.post('http://127.0.0.1:8000/', data=new_d, verify=False)
print(response)
print(response.json)
print(response.json())
