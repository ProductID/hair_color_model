"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from django.conf.urls.static import static

from myapp import views
from mysite import settings
print('=============================')
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.model_form_upload, name='home'),
    path('hair-colour', views.new_hair_colour, name='new_hair_colour'),
    path('age-prediction', views.age_prediction, name='age_prediction'),
    path('ethnicity', views.race_prediction, name='race_prediction'),

    path('hair', views.hair_mod, name='hair'),
    path('start-free-trial', views.start_free_trial, name='start_free_trial'),
    path('activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$', views.activate, name='activate'),
    path('login', views.login, name='login'),
    path('forgot-password', views.forgot_password, name='forgot_password'),

]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
