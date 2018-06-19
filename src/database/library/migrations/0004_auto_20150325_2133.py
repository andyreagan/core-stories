# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0003_auto_20150325_2132'),
    ]

    operations = [
        migrations.AlterField(
            model_name='book',
            name='exclude',
            field=models.BooleanField(default=False),
            preserve_default=True,
        ),
    ]
