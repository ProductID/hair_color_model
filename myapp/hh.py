import os.path
import os
from django.contrib.auth import authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.contrib.sites.shortcuts import get_current_site
from django.core.files import File
from django.core.mail import EmailMultiAlternatives
from django.shortcuts import render, redirect
from django.template.loader import get_template
from django.contrib import messages
# from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
# from jwt.utils import force_bytes
from jwt.utils import force_bytes

from myapp.form import DocumentForm, Startfreetrial

# from myapp.stargan2.new_main_file import add_with_front_panel
from myapp.stargan2.finall_lips import main
from myapp.stargan2.hair_final import main_hair
# image_=main(image_path,color_code)
from myapp.tokens import account_activation_token
import settings
from settings import BASE_DIR
from .models import *


def model_form_upload(request):
    # tf.keras.backend.clear_session()
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            src_image_in = request.FILES.get("src_image")
            form_1 = form.save()
            if src_image_in is None:
                src_image_slider = request.POST.get("src_image_slider")
                print(src_image_slider, "2-src-img")

                if len(src_image_slider) <= 0:
                    src_image_slider = "/static/input_images/7_we8vSID.jpg"

                print(src_image_slider, "^^^^^^^^^^&&&&&&&&&&&&&&&&&*****************", len(src_image_slider))

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
            text_path = src_image.split(".")
            text_path_new = text_path[0]
            image_download = text_path_new.split('/')
            download_mat = str(image_download[-1]) + '.jpg'
            # save_path = str('static/out/')+download_mat
            save_path = str('/media/')+download_mat
            if ref_image_2.split('/')[-1]=='COLORCODE.png':
                color_code = {"RED": (0, 0, 190)}
            elif ref_image_2.split('/')[-1]=='COLORCODE2.png':
                color_code = {"ROSE_GOLD": (121, 110, 183)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE3.png':
                color_code = {"MAROON": (0, 0, 128)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE4.png':
                color_code = {"REDWOOD": (82, 90, 164)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE5.png':
                color_code = {"RASPBERRY": (60, 31, 210)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE6.png':#RGB (141,2,31)  --BURGUNDY
                color_code = {"BURGUNDY": (31, 2, 141)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE7.png': # RGB (247,52,122) --ROSEBONBON
                color_code = {"ROSEBONBON": (122, 52, 247)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE8.png': #RGB (124,10,2)   --BARN RED
                color_code = {"BARN_RED": (2, 10, 124)}
            elif ref_image_2.split('/')[-1] == 'COLORCODE9.png': #RGB (150,0,25) --CARMINE
                color_code = {"CARMINE": (25, 0, 150)}
            else:
                color_code = {"vamptastic_plum": (0, 104, 255)}

            main(src_image, color_code,save_path)

    else:
        form = DocumentForm()
    return render(request, 'demo.html', locals())

def hair_mod(request):
    # tf.keras.backend.clear_session()
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            src_image_in = request.FILES.get("src_image")
            print(src_image_in,'ddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
            form_1 = form.save()
            if src_image_in is None:
                src_image_slider = request.POST.get("src_image_slider")
                d_path = os.path.join(BASE_DIR)
                img_path=f"{d_path}{src_image_slider}"
                f = open(img_path, 'rb')
                form.src_image= File(f)
                src_image = img_path
            else:
                src_image = form_1.src_image.path
            src_image_1 = form_1.src_image
            # ref_image = form_1.ref_image.path
            ref_image = request.POST.get('ref_image', '')
            ref_image_1 = ref_image
            ref_image_2 = str(BASE_DIR) + '/' + str(ref_image)
            text_path = src_image.split(".")
            text_path_new = text_path[0]
            image_download = text_path_new.split('/')
            download_mat = str(image_download[-1]) + '.jpg'
            save_path = str('static/out/')+download_mat
            if ref_image_2.split('/')[-1]=='COLORCODE.png':
                color_code = [0, 0, 190]
            elif ref_image_2.split('/')[-1]=='COLORCODE2.png':
                color_code = [121, 110, 183]
            elif ref_image_2.split('/')[-1] == 'COLORCODE3.png':
                color_code = [0, 0, 128]
            elif ref_image_2.split('/')[-1] == 'COLORCODE4.png':
                color_code = [82, 90, 164]
            elif ref_image_2.split('/')[-1] == 'COLORCODE5.png':
                color_code = [60, 31, 210]
            elif ref_image_2.split('/')[-1] == 'COLORCODE6.png':#RGB (141,2,31)  --BURGUNDY
                color_code = [31, 2, 141]
            elif ref_image_2.split('/')[-1] == 'COLORCODE7.png': # RGB (247,52,122) --ROSEBONBON
                color_code = [122, 52, 247]
            elif ref_image_2.split('/')[-1] == 'COLORCODE8.png': #RGB (124,10,2)   --BARN RED
                color_code = [2, 10, 124]
            elif ref_image_2.split('/')[-1] == 'COLORCODE9.png': #RGB (150,0,25) --CARMINE
                color_code = [25, 0, 150]
            else:
                color_code = [0, 104, 255]
            main_hair(src_image, color_code,save_path)

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
            firstname,after_domain=to_email.split('@')
            user.username=firstname
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


def register_send_mails(emails,username):
    d = ({'email': emails,'username':username})
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


def activate(request, uidb64, token,backend='django.contrib.auth.backends.ModelBackend'):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user,backend='django.contrib.auth.backends.ModelBackend')
        register_send_mails(request.user.email, request.user.username)
        return redirect('/post_property')
    else:
        return render(request, 'account_verification/account_activation_invalid.html', locals())

def account_activation_sent(request):
    return render(request, 'account_verification/account_activation_sent.html', locals())