# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Book',
            fields=[
                ('id', models.AutoField(serialize=False, verbose_name='ID', primary_key=True, auto_created=True)),
                ('filename', models.CharField(max_length=100)),
                ('title', models.CharField(max_length=200)),
                ('author', models.CharField(max_length=200)),
                ('language', models.CharField(max_length=100)),
                ('happs', models.FloatField()),
                ('length', models.IntegerField()),
                ('ignorewords', models.CharField(max_length=400)),
                ('wiki', models.URLField()),
            ],
            options={
                'ordering': ('author',),
            },
            bases=(models.Model,),
        ),
    ]
