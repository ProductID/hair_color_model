import os.path
import os
from django.contrib.auth import authenticate
from django.contrib.auth.forms import AuthenticationForm
# from django.contrib.auth.models import User
from django.contrib.sites.shortcuts import get_current_site
from django.core.files import File
from django.core.mail import EmailMultiAlternatives
from django.http import HttpResponse,HttpResponseRedirect
# from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.template.loader import get_template
from django.contrib import messages
from django.utils.encoding import force_bytes
# from django.urls import reverse
# from django.utils.encoding import force_bytes, force_text
# from django.utils.encoding import force_text
from django.utils.http import urlsafe_base64_encode
# from jwt.utils import force_bytes
# from jwt.utils import force_bytes

from myapp.form import DocumentForm, Startfreetrial
# import pandas as pd
import requests

# from myapp.models import Startfree
# from myapp.stargan2.new_main_file import add_with_front_panel
from myapp.stargan2.finall_lips import main
# image_=main(image_path,color_code)
from myapp.tokens import account_activation_token
from mysite import settings
from mysite.settings import BASE_DIR
from .models import *
from .stargan2.hair_final import main_hair
from .give_data import *
# from Barbershop import main_c_new
from .age_race_prediction import age_predictor, race_predictor
import matplotlib.pyplot as plt
from multiprocessing import Process
from threading import Thread
import ray
import threading
def age_prediction(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            src_image_in = request.FILES.get("src_image")
            form_1 = form.save()
            if src_image_in is None:
                src_image_slider = request.POST.get("src_image_slider")
                print(src_image_slider, "2-src-img", len(src_image_slider), "aaaaa")

                if len(src_image_slider) <= 0:
                    src_image_slider = "/static/input_images/7_we8vSID.jpg"

                print(src_image_slider, "aaaaa", len(src_image_slider), "aaaaa")
                d_path = os.path.join(BASE_DIR)
                img_path = f"{d_path}{src_image_slider}"
                f = open(img_path, 'rb')
                print(File(f), "]]]]]]]]]", type(File(f)))
                form.src_image = File(f)
                src_image = img_path

            else:
                src_image = form_1.src_image.path

            d_path = os.path.join(BASE_DIR)
            print(src_image, '----------------------mmmmm')
            # print(type(src_image),'----------------------mmmmm')
            img, imgname = age_predictor(src_image)
            # print(imgname,"------")
            # print(img,"----img--")
            imggggg = f'/media/{imgname}'
            image_to_save = f"{d_path}/media/{imgname}"
            print(image_to_save, "---------------imggggggg", )
            plt.imsave(image_to_save, img)
            return render(request, 'age_prediction.html', locals())

    else:
        form = DocumentForm()
    return render(request, 'age_prediction.html', locals())


def race_prediction(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            src_image_in = request.FILES.get("src_image")
            form_1 = form.save()
            if src_image_in is None:
                src_image_slider = request.POST.get("src_image_slider")
                print(src_image_slider, "2-src-img", len(src_image_slider), "aaaaa")

                if len(src_image_slider) <= 0:
                    src_image_slider = "/static/input_images/7_we8vSID.jpg"

                print(src_image_slider, "aaaaa", len(src_image_slider), "aaaaa")
                d_path = os.path.join(BASE_DIR)
                img_path = f"{d_path}{src_image_slider}"
                f = open(img_path, 'rb')
                print(File(f), "]]]]]]]]]", type(File(f)))
                form.src_image = File(f)
                src_image = img_path

            else:
                src_image = form_1.src_image.path

            d_path = os.path.join(BASE_DIR)
            print(src_image, '----------------------mmmmm')
            # print(type(src_image),'----------------------mmmmm')
            img, imgname = race_predictor(src_image)
            # print(imgname,"------")
            # print(img,"----img--")
            imggggg = f'/media/{imgname}'
            image_to_save = f"{d_path}/media/{imgname}"
            print(image_to_save, "---------------imggggggg", )
            plt.imsave(image_to_save, img)
            return render(request, 'race_prediction.html', locals())

    else:
        form = DocumentForm()
    return render(request, 'race_prediction.html', locals())

#
@ray.remote
def getHairResults(new_d):
    print(new_d)
    print("hitting api")
    requests.post('http://localhost:8000/hair_color_api', data=new_d, verify=False)

@ray.remote
def goToHairResults():
    print("mmmmm")
    # return HttpResponseRedirect("/hair")

    return redirect('/hair-colour')



def new_hair_colour(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            src_image_in = request.FILES.get("src_image")
            print(src_image_in, 'ddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
            form_1 = form.save()
            if src_image_in is None:
                src_image_slider = request.POST.get("src_image_slider")
                print(src_image_slider, "2-src-img", len(src_image_slider), "aaaaa")

                if len(src_image_slider) <= 0:
                    src_image_slider = "/static/input_images/7_we8vSID.jpg"

                print(src_image_slider, "aaaaa", len(src_image_slider), "aaaaa")
                d_path = os.path.join(BASE_DIR)
                img_path = f"{d_path}{src_image_slider}"
                f = open(img_path, 'rb')
                print(File(f), "]]]]]]]]]", type(File(f)))
                form.src_image = File(f)
                src_image = img_path

            else:
                src_image = form_1.src_image.path
            src_image_1 = form_1.src_image
            print(src_image_1, "----------------1")
            print(src_image, "----------------")
            # ref_image = form_1.ref_image.path
            ref_image = request.POST.get('ref_image', '')
            ref_image_1 = ref_image
            ref_image_2 = str(BASE_DIR) + str(ref_image)
            print(ref_image_2, '000000000000000000000000000')
            text_path = src_image.split(".")
            text_path_new = text_path[0]
            image_download = text_path_new.split('/')
            download_mat = str(image_download[-1]) + '.jpg'
            save_path = str(BASE_DIR) + str('/media/input') + download_mat
            p_imagepath = str('/media/input') + download_mat
            # if ref_image_2.split('/')[-1]=='57.png':
            #     ref_image = '57_blonde.png'
            ref_img = 'blonde_ref.png'
            # ref_image = '916.jpg'
            # ref_img_lis = [ '4.png','40.png','black_final.png','black_finalo.png']
            # for ref_img in ref_img_lis:
            #     print(ref_img,"------------ref_image_from loop")
            # elif ref_image_2.split('/')[-1]=='58.png':
            #     ref_image = '58_blonde.png'
            # elif ref_image_2.split('/')[-1] == '915.jpg':
            #     ref_image = '915_blonde.png'
            # else:
            #     ref_image = '57.png'
            #     new_d = {'input_dir': 'media/input', 'output_dir': 'media/output', 'im_path1': src_image,
            #              'im_path2': src_image,
            #              'im_path3': ref_img,
            #              'sign': 'realistic', 'smooth': 5, 'size': 1024, 'ckpt': '/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/celebrity_model-master/Barbershop/pretrained_models/ffhq.pt',
            #              'channel_multiplier': 2, 'latent': 512, 'n_mlp': 8,
            #              'device': 'cuda', 'seed': None, 'tile_latent': None, 'opt_name': 'adam', 'learning_rate': 0.01,
            #              'lr_schedule': 'fixed',
            #              'save_intermediate': None, 'save_interval': 300, 'verbose': None, 'seg_ckpt': '/media/aj/9c4728aa-3a45-44cf-ada0-079baa4684ac/home/webtunix/celebrity_model-master/Barbershop/pretrained_models/seg.pth',
            #              'percept_lambda': 1.0,
            #              'l2_lambda': 1.0, 'p_norm_lambda': 0.001, 'l_F_lambda': 0.1, 'W_steps': 1100, 'FS_steps': 250,
            #              'ce_lambda': 1.0, 'style_lambda': 4e4,
            #              'align_steps1': 140, 'align_steps2': 100, 'face_lambdat': 1.0, 'hair_lambda': 1.0, 'blend_steps': 40}
            #
            #     ddd=giveData(new_d)
            #     output_image_path=main_c_new.main(ddd)
            #     print(output_image_path,"----------output-image")

            new_d = {'input_dir': 'media/input', 'output_dir': 'media/output', 'im_path1': src_image,
                     'im_path2': src_image,
                     'im_path3': ref_img,
                     'sign': 'realistic', 'smooth': 5, 'size': 1024,
                     'ckpt': 'Barbershop/pretrained_models/ffhq.pt',
                     'channel_multiplier': 2, 'latent': 512, 'n_mlp': 8,
                     'device': 'cuda', 'seed': None, 'tile_latent': None, 'opt_name': 'adam', 'learning_rate': 0.01,
                     'lr_schedule': 'fixed',
                     'save_intermediate': None, 'save_interval': 300, 'verbose': None,
                     'seg_ckpt': 'Barbershop/pretrained_models/seg.pth',
                     'percept_lambda': 1.0,
                     'l2_lambda': 1.0, 'p_norm_lambda': 0.001, 'l_F_lambda': 0.1, 'W_steps': 1100, 'FS_steps': 250,
                     'ce_lambda': 1.0, 'style_lambda': 4e4,
                     'align_steps1': 140, 'align_steps2': 100, 'face_lambdat': 1.0, 'hair_lambda': 1.0,
                     'blend_steps': 40}
            # response = requests.post(url='<your_url_here>', params=payload)


            # p1 = Process(target=getHairResults(new_d))
            # p1.start()
            # p2 = Process(target=goToHairResults)
            # p2.start()
            # p1 = threading.Thread(target=getHairResults(new_d))
            # p2 = threading.Thread(target=goToHairResults)
            # p1.start();
            # p2.start()
            # Thread(target=getHairResults(new_d)).start()
            # print("after hitting 1st")
            # Thread(target=goToHairResults).start()
            # @ray.remote
            # def getHairResults():
            #     global new_d
            #     print(new_d)
            #     print("hitting api")
            #     requests.post('http://localhost:8000/hair_color_api', data=new_d, verify=False)

            # getHairResults()
            # ray.init()
            # ray.get([getHairResults.remote(new_d), goToHairResults.remote()])
            #
            #
            # # getHairResults(new_d)
            # print("after post requests")
            res=requests.post('http://localhost:8000/hair_color_api', data=new_d, verify=False)
            print(res,"-------------1")
            print(res.json,"---------2")
            print(res.json(),"--------3")
            result=res.json()
            output_image_path=result.get("output_image_path",None)


            return render(request, 'hair_colour.html', locals())


    else:
        form = DocumentForm()

    return render(request, 'hair_colour.html', locals())


def model_form_upload(request):
    # tf.keras.backend.clear_session()
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            print(form, '+++++0000')
            src_image_in = request.FILES.get("src_image")
            print(src_image_in, "1-src-//////----img", type(src_image_in))
            form_1 = form.save()

            if src_image_in is None:
                src_image_slider = request.POST.get("src_image_slider")
                print(src_image_slider, "2-src-img", len(src_image_slider), "aaaaa")

                if len(src_image_slider) <= 0:
                    src_image_slider = "/static/input_images/7_we8vSID.jpg"

                print(src_image_slider, "aaaaa", len(src_image_slider), "aaaaa")
                d_path = os.path.join(BASE_DIR)
                img_path = f"{d_path}{src_image_slider}"
                f = open(img_path, 'rb')
                print(File(f), "]]]]]]]]]", type(File(f)))
                form.src_image = File(f)
                src_image = img_path

            else:
                src_image = form_1.src_image.path
            # if src_image is None:
            #     print("yyy")
            #     src_image_slider = request.POST.get("src_image_slider")
            #     print(src_image_slider, "2-src-img")
            #
            #     # src_image = src_image_slider
            #     # print(src_image, "1-src---------------------img", type(src_image))
            #     #
            #     doc_obj=Document(id=form_1.id)
            #     print(doc_obj.src_image,'88')
            #     # f = open(src_image_slider, 'r')
            #     d_path = os.path.join(BASE_DIR)
            #     img_path=f"{d_path}{src_image_slider}"
            #     f = open(img_path, 'rb')
            #     print(File(f),"]]]]]]]]]",type(File(f)))
            #     doc_obj.src_image = File(f)
            #     # doc_obj.src_image=src_image
            #     print(doc_obj.src_image, '------99')
            #     doc_obj.save()
            #
            #
            #
            #
            #
            # else:
            #     src_image = form_1.src_image.path

            # src_image = form_1.src_image.path
            src_image_1 = form_1.src_image
            # ref_image = form_1.ref_image.path
            ref_image = request.POST.get('ref_image', '')
            ref_image_1 = ref_image
            ref_image_2 = str(BASE_DIR) + '/' + str(ref_image)
            print(ref_image_2, '000000000000000000000000000')
            text_path = src_image.split(".")
            text_path_new = text_path[0]
            image_download = text_path_new.split('/')
            download_mat = str(image_download[-1]) + '.jpg'
            save_path = str(BASE_DIR) + str('/media/') + download_mat
            p_imagepath = str('/media/') + download_mat

            '''
            RGB (211,0,0)        -- RED
RGB (183,110,121)  -- ROSE GOLD
RGB (128,0,0)     -- MAROON

RGB  (164,90,82)  -- REDWOOD

RGB (210,31,60)  --RASPBERRY
RGB (141,2,31)  --BURGUNDY
RGB (247,52,122) --ROSEBONBON
RGB (124,10,2)   --BARN RED

RGB (150,0,25) --CARMINE



            '''
            if ref_image_2.split('/')[-1] == 'COLORCODE.png':
                color_code = {"RED": (0, 0, 190)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE2.png':
                color_code = {"ROSE_GOLD": (121, 110, 183)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE3.png':
                color_code = {"MAROON": (0, 0, 128)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE4.png':
                color_code = {"REDWOOD": (82, 90, 164)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE5.png':
                color_code = {"RASPBERRY": (60, 31, 210)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE6.png':  # RGB (141,2,31)  --BURGUNDY
                color_code = {"BURGUNDY": (31, 2, 141)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE7.png':  # RGB (247,52,122) --ROSEBONBON
                color_code = {"ROSEBONBON": (122, 52, 247)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE8.png':  # RGB (124,10,2)   --BARN RED
                color_code = {"BARN_RED": (2, 10, 124)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE9.png':  # RGB (150,0,25) --CARMINE
                color_code = {"CARMINE": (25, 0, 150)}
            else:
                color_code = {"vamptastic_plum": (0, 104, 255)}

            # input('dddddddddddddddddddddddddddddddddddddddddddddd')

            print(save_path, 'paaaathhh333')
            print('lllllllllllllllllllllllllllllllllllllllllllllllllll')
            # add_with_front_panel(src_image, ref_image_2, save_path)
            # color_code={"vamptastic_plum": (115, 104, 255)}
            main(src_image, color_code, save_path)

    else:
        form = DocumentForm()
    return render(request, 'demo.html', locals())


def hair_mod(request):
    # tf.keras.backend.clear_session()
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            src_image_in = request.FILES.get("src_image")
            print(src_image_in, 'ddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
            form_1 = form.save()
            if src_image_in is None:
                src_image_slider = request.POST.get("src_image_slider")
                print(src_image_slider, "2-src-img", len(src_image_slider), "aaaaa")

                if len(src_image_slider) <= 0:
                    src_image_slider = "/static/input_images/7_we8vSID.jpg"

                print(src_image_slider, "aaaaa", len(src_image_slider), "aaaaa")
                d_path = os.path.join(BASE_DIR)
                img_path = f"{d_path}{src_image_slider}"
                f = open(img_path, 'rb')
                print(File(f), "]]]]]]]]]", type(File(f)))
                form.src_image = File(f)
                src_image = img_path

            else:
                src_image = form_1.src_image.path
            src_image_1 = form_1.src_image
            # ref_image = form_1.ref_image.path
            ref_image = request.POST.get('ref_image', '')
            ref_image_1 = ref_image
            ref_image_2 = str(BASE_DIR) + '/' + str(ref_image)
            print(ref_image_2, '000000000000000000000000000')
            text_path = src_image.split(".")
            text_path_new = text_path[0]
            image_download = text_path_new.split('/')
            download_mat = str(image_download[-1]) + '.jpg'
            save_path = str(BASE_DIR) + str('/media/') + download_mat
            p_imagepath = str('/media/') + download_mat
            if ref_image_2.split('/')[-1] == 'COLORCODE.png':
                color_code = [0, 0, 190]
            elif ref_image_2.split('/')[-1] == 'COLORCODE2.png':
                color_code = [121, 110, 183]
            elif ref_image_2.split('/')[-1] == 'COLORCODE3.png':
                color_code = [0, 0, 128]
            elif ref_image_2.split('/')[-1] == 'COLORCODE4.png':
                color_code = [82, 90, 164]
            elif ref_image_2.split('/')[-1] == 'COLORCODE5.png':
                color_code = [60, 31, 210]
            elif ref_image_2.split('/')[-1] == 'COLORCODE6.png':  # RGB (141,2,31)  --BURGUNDY
                color_code = [31, 2, 141]
            elif ref_image_2.split('/')[-1] == 'COLORCODE7.png':  # RGB (247,52,122) --ROSEBONBON
                color_code = [122, 52, 247]
            elif ref_image_2.split('/')[-1] == 'COLORCODE8.png':  # RGB (124,10,2)   --BARN RED
                color_code = [2, 10, 124]
            elif ref_image_2.split('/')[-1] == 'COLORCODE9.png':  # RGB (150,0,25) --CARMINE
                color_code = [25, 0, 150]
            else:
                color_code = [0, 104, 255]
            main_hair(src_image, color_code, save_path)

    else:
        form = DocumentForm()
    return render(request, 'demo.html', locals())


def start_free_trial(request):
    if request.method == 'POST':
        form = Startfreetrial(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.is_seller = True
            to_email = form.cleaned_data.get('email')
            firstname, after_domain = to_email.split('@')
            user.username = firstname
            user.save()

            current_site = get_current_site(request)
            d = ({
                'user': user,
                'domain': current_site.domain,
                # 'uid': urlsafe_base64_encode(force_bytes(user.pk)).decode(),
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })
            plaintext = get_template('email.txt')
            htmly = get_template(
                'Email.html')
            subject, from_email, to = 'Accoount Verification email', settings.DEFAULT_FROM_EMAIL, to_email
            text_content = plaintext.render(d)
            html_content = htmly.render(d)
            msg = EmailMultiAlternatives(
                subject, text_content, from_email, [to])
            msg.attach_alternative(html_content, "text/html")
            msg.send()

            return redirect('account_activation_sent')
        else:
            form = Startfreetrial(request.POST)
    else:
        form = Startfreetrial()
    return render(request, 'start-free-trial.html')


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            form = login(request, user)
            messages.success(request, f'Welcome {username} !!')
            return redirect('demo')
        else:
            messages.info(request, f'account done not exist plz sign up')
    form = AuthenticationForm()
    return render(request, 'login.html', {'form': form, 'title': 'log in'})


def register_send_mails(emails, username):
    d = ({'email': emails, 'username': username})
    plaintext = get_template('email.txt')
    htmly = get_template('welcome_template.html')
    subject, from_email, to = "Welcome To Celebrity model", settings.DEFAULT_FROM_EMAIL, emails
    text_content = plaintext.render(d)
    html_content = htmly.render(d)
    msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
    msg.attach_alternative(html_content, "text/html")
    msg.send()


def forgot_password(request):
    return render(request, 'password/forgot-password.html')


def activate(request, uidb64, token, backend='django.contrib.auth.backends.ModelBackend'):
    # try:
    #     uid = force_text(urlsafe_base64_decode(uidb64))
    #     user = User.objects.get(pk=uid)
    # except (TypeError, ValueError, OverflowError, User.DoesNotExist):
    #     user = None
    # # if user is not None and account_activation_token.check_token(user, token):
    #     user.is_active = True
    #     user.save()
    #     login(request, user,backend='django.contrib.auth.backends.ModelBackend')
    #     register_send_mails(request.user.email, request.user.username)
    #     return redirect('/post_property')
    # else:
    return render(request, 'account_verification/account_activation_invalid.html', locals())


def account_activation_sent(request):
    return render(request, 'account_verification/account_activation_sent.html', locals())
