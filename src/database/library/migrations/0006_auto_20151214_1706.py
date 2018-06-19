# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0005_book_numuniqwords'),
    ]

    operations = [
        migrations.RenameField(
            model_name='book',
            old_name='happsEnd',
            new_name='happs_end',
        ),
        migrations.RenameField(
            model_name='book',
            old_name='happsMax',
            new_name='happs_max',
        ),
        migrations.RenameField(
            model_name='book',
            old_name='happsMin',
            new_name='happs_min',
        ),
        migrations.RenameField(
            model_name='book',
            old_name='happsStart',
            new_name='happs_start',
        ),
        migrations.RenameField(
            model_name='book',
            old_name='happsVariance',
            new_name='happs_variance',
        ),
        migrations.AddField(
            model_name='book',
            name='scaling_exponent',
            field=models.FloatField(default=-1.0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='book',
            name='scaling_exponent_top100',
            field=models.FloatField(default=-1.0),
            preserve_default=False,
        ),
    ]
