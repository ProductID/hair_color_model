# Generated by Django 3.1.1 on 2022-07-27 12:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0005_login'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='src_image',
            field=models.FileField(blank=True, null=True, upload_to='src_image/'),
        ),
    ]
