# Generated by Django 2.1.7 on 2019-03-19 16:25

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataSet',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('identif', models.IntegerField()),
                ('nombre', models.CharField(max_length=20)),
                ('cantAtributos', models.IntegerField()),
            ],
        ),
    ]
