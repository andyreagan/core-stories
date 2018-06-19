# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0004_auto_20150325_2133'),
    ]

    operations = [
        migrations.AddField(
            model_name='book',
            name='numUniqWords',
            field=models.IntegerField(default=0.0),
            preserve_default=False,
        ),
    ]
