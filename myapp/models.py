from django.contrib.auth.models import User
from django.db import models

# Create your models here.
from django.db.models.signals import post_save
from django.dispatch import receiver


class Document(models.Model):
    # src_image = models.FileField(upload_to='src_image/')
    src_image = models.FileField(upload_to='src_image/', blank=True,null=True)
    # ref_image = models.FileField(upload_to='ref_image/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Startfree(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    phone = models.CharField(max_length=100)
    password = models.CharField(max_length=100)


class login(models.Model):
    email = models.EmailField(max_length=100)
    password = models.CharField(max_length=100)