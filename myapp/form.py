from django import forms
from django.contrib.auth.forms import UserCreationForm

from myapp.models import Document, Startfree, login
from django.contrib.auth.models import User

class DocumentForm(forms.ModelForm):
    src_image= forms.FileField(required=False)
    class Meta:
        model = Document
        fields = ['src_image']


class Startfreetrial(forms.ModelForm):
    name = forms.CharField(required=True, max_length=100)
    email = forms.EmailField(required=True, max_length=100)
    phone = forms.CharField(required=True, max_length=100)
    password = forms.CharField(required=True, max_length=100)
    class Meta:
        model = Startfree
        fields = ['name', 'email', 'phone', 'password']


class login(forms.ModelForm):
    email = forms.EmailField(required=True, max_length=100)
    password = forms.CharField(required=True, max_length=100)
    class Meta:
        model = login
        fields = ['email', 'password']
