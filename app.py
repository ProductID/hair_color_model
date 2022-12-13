import torch
import os
from Barbershop.models.Embedding import Embedding
from Barbershop.models.Alignment import Alignment
from Barbershop.models.Blending import Blending
import pysftp
# from PIL import Image
# import urllib.request
# import requests
# from io import BytesIO
# import io
# from PIL import Image


def init():
    device = 0 if torch.cuda.is_available() else -1
    checkpoint = torch.load('Barbershop/pretrained_models/ffhq.pt')
    checkpoint_sh = torch.load('Barbershop/pretrained_models/seg.pth')


def inference(args):
    ii2s = Embedding(args)

    im_path1 = os.path.join(args.input_dir, args.im_path1)
    im_path2 = os.path.join(args.input_dir, args.im_path2)
    im_path3 = os.path.join(args.input_dir, args.im_path3)

    # im_path1 = args.im_path1
    # im_path2 = args.im_path2
    # im_path3 = args.im_path3

    # response__ = requests.get(im_path1)
    # response__ = requests.get("https://www.google.com/imgres?imgurl=https%3A%2F%2Fimg.freepik.com%2Ffree-photo%2Fportrait-white-man-isolated_53876-40306.jpg&imgrefurl=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Fman-face&tbnid=JLYULrt3NjO83M&vet=12ahUKEwiArKOb5tD7AhU6BLcAHekXC9YQMygBegUIARDkAQ..i&docid=4JvHjCMKYtVnyM&w=626&h=487&q=face%20image&client=ubuntu&ved=2ahUKEwiArKOb5tD7AhU6BLcAHekXC9YQMygBegUIARDkAQ")
    # im_path1 = Image.open(io.BytesIO(response__.content))
    #
    # response__ = requests.get(im_path2)
    # im_path2 = Image.open(io.BytesIO(response__.content))
    #
    # response__ = requests.get(im_path3)
    # im_path3 = Image.open(io.BytesIO(response__.content))
    # with urllib.request.urlopen(im_path1) as url1:
    #     im_path1 = Image.open(url1)
    # with urllib.request.urlopen(im_path2) as url2:
    #     im_path2 = Image.open(url2)
    # with urllib.request.urlopen(im_path3) as url3:
    #     im_path3 = Image.open(url3)

    im_set = {im_path1, im_path2, im_path3}
    ii2s.invert_images_in_W([*im_set])
    ii2s.invert_images_in_FS([*im_set])

    align = Alignment(args)
    align.align_images(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)
    if im_path2 != im_path3:
        align.align_images(im_path1, im_path3, sign=args.sign, align_more_region=False, smooth=args.smooth,
                           save_intermediate=False)

        # blend = Blending(args)
        # output_image_path=blend.blend_images(im_path1, im_path2, im_path3, sign=args.sign)
        # return output_image_path
    blend = Blending(args)
    output_image_path = blend.blend_images(im_path1, im_path2, im_path3, sign=args.sign)
    print(im_path1, 'llll------------------------1')
    print(im_path2, 'llll------------------------------2')
    print(im_path3, 'llll---------------------------3')
    print(output_image_path, 'llll---------------------4')

    # unique name pending-----------------------------------


    srv = pysftp.Connection(host="3.70.151.70", username="ubuntu", password="sd9809$%^")

    #  upload file
    with srv.cd('/home/ubuntu/uploadimages'):  # chdir to public
        srv.put(output_image_path)  # upload file to nodejs/

    # download file
    # srv.get("/home/ubuntu/uploadimages/christopher-campbell-rDEOVtE7vOs-unsplash.jpg",
    #         "christopher-campbell-rDEOVtE7vOs-unsplash.jpg")

    # Closes the connection
    srv.close()

    return output_image_path
