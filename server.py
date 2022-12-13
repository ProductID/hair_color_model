
from fastapi import FastAPI, File, UploadFile,Request
from fastapi.encoders import jsonable_encoder
import app as u_app
from app import inference
from give_data_fast import giveData
import uvicorn
from datetime import datetime


now = datetime.now()

print('now=@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',now)

from fastapi.responses import JSONResponse
import os
import multiprocessing
from multiprocessing import set_start_method
app = FastAPI()
try:
    set_start_method('spawn')
except:
    pass


def print_square(num,ddd):
        """
        function to print square of given num
        """
        # print("Square: {}".format(num * num)
        # global ddd
        # set_start_method('spawn')
        output_image_path = u_app.inference(ddd)
        return 'im done'

@app.post("/")
async def inference(request: Request):
    new_d = await request.json()
    print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv',new_d)
    # print(da)
    new_d = jsonable_encoder(new_d)
    input_dir = str(new_d.get("input_dir", "Barbershop/input/face"))
    output_dir = str(new_d.get("output_dir", "Barbershop/output_dir"))
    im_path1 = str(new_d.get("im_path1", "18.png"))
    im_path2 = str(new_d.get("im_path2", "18.png"))
    im_path3 = str(new_d.get("im_path3", "11.png"))
    sign = str(new_d.get("sign", 'realistic'))
    smooth = int(new_d.get("smooth", 5))
    size = int(new_d.get("size", 1024))
    ckpt = str(new_d.get("ckpt", "Barbershop/pretrained_models/ffhq.pt"))
    channel_multiplier = int(new_d.get("channel_multiplier", 2))
    latent = int(new_d.get("latent", 512))
    n_mlp = int(new_d.get("n_mlp", 8))
    device = str(new_d.get("device", "cuda"))
    seed = str(new_d.get("seed", None))
    tile_latent = (new_d.get("tile_latent", None))
    opt_name = str(new_d.get("opt_name", "adam"))
    learning_rate = float(new_d.get("learning_rate", 0.01))
    lr_schedule = str(new_d.get("lr_schedule", "fixed"))
    save_intermediate = (new_d.get("save_intermediate", None))
    save_interval = int(new_d.get("save_interval", 300))
    verbose = (new_d.get("verbose", None))
    seg_ckpt = str(new_d.get("seg_ckpt", "Barbershop/pretrained_models/seg.pth"))
    percept_lambda = float(new_d.get("percept_lambda", 1.0))
    l2_lambda = float(new_d.get("l2_lambda", 1.0))
    p_norm_lambda = float(new_d.get("p_norm_lambda", 0.001))
    l_F_lambda = float(new_d.get("l_F_lambda", 0.1))
    W_steps = int(new_d.get("W_steps", 1100))
    FS_steps = int(new_d.get("FS_steps", 250))
    ce_lambda = float(new_d.get("ce_lambda", 1.0))
    style_lambda = float(new_d.get("style_lambda", 40000.0))
    align_steps1 = int(new_d.get("align_steps1", 140))
    align_steps2 = int(new_d.get("align_steps2", 100))
    face_lambdat = float(new_d.get("face_lambdat", 1.0))
    hair_lambda = float(new_d.get("hair_lambda", 1.0))
    blend_steps = int(new_d.get("blend_steps", 40))
    new_d = {'input_dir': input_dir, 'output_dir': output_dir, 'im_path1': im_path1,
                'im_path2': im_path2,
                'im_path3': im_path3,
                'sign': sign, 'smooth': smooth, 'size': size, 'ckpt': ckpt,
                'channel_multiplier': channel_multiplier, 'latent': latent, 'n_mlp': n_mlp,
                'device': device, 'seed': seed, 'tile_latent': tile_latent, 'opt_name': opt_name,
                'learning_rate': learning_rate,
                'lr_schedule': lr_schedule,
                'save_intermediate': save_intermediate, 'save_interval': save_interval, 'verbose': verbose,
                'seg_ckpt': seg_ckpt,
                'percept_lambda': percept_lambda,
                'l2_lambda': l2_lambda, 'p_norm_lambda': p_norm_lambda, 'l_F_lambda': l_F_lambda, 'W_steps': W_steps,
                'FS_steps': FS_steps,
                'ce_lambda': ce_lambda, 'style_lambda': style_lambda,
                'align_steps1': align_steps1, 'align_steps2': align_steps2, 'face_lambdat': face_lambdat,
                'hair_lambda': hair_lambda, 'hair_lambda': hair_lambda, 'blend_steps': blend_steps}

    print(input_dir,"------------------")
    global ddd
    ddd = giveData(new_d)

    print("now =##################################", now)
    print(ddd)
    p1 = multiprocessing.Process(target=print_square, args=(10, ddd))
    p1.start()

    # output_image_path = main_new.main(ddd)
    output_image_path = os.path.join(new_d['output_dir'],
                                     '{}_{}_{}_{}.png'.format(new_d['im_path1'], new_d['im_path2'], new_d['im_path3'],
                                                              new_d['sign']))

    print(output_image_path, "----------output-image")
    mdata = {"output_image_path": output_image_path}
    return JSONResponse(content=mdata)

if __name__ == "__main__":
    uvicorn.run("server:app", port=8000)




