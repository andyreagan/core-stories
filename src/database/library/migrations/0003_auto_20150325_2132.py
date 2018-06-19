# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0002_auto_20150325_2130'),
    ]

    operations = [
        migrations.AddField(
            model_name='book',
            name='exclude',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='book',
            name='excludeReason',
            field=models.CharField(max_length=100, blank=True, null=True),
            preserve_default=True,
        ),
    ]
