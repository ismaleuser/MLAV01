# Generated by Django 2.1.7 on 2019-04-16 16:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('modelos', '0003_dataset_uploadfile'),
    ]

    operations = [
        migrations.CreateModel(
            name='Algoritmo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nombre', models.CharField(max_length=50)),
                ('info', models.CharField(max_length=250)),
            ],
        ),
        migrations.CreateModel(
            name='Modelo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nombre', models.CharField(max_length=50)),
                ('direccion', models.CharField(max_length=250)),
                ('fechaGenerado', models.DateTimeField(blank=True, null=True)),
                ('algoritmo', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='modelos.Algoritmo')),
            ],
        ),
        migrations.RenameField(
            model_name='caso',
            old_name='clase',
            new_name='claseS',
        ),
        migrations.RemoveField(
            model_name='caso',
            name='clasificador',
        ),
        migrations.RemoveField(
            model_name='clasificador',
            name='base_caso',
        ),
        migrations.RemoveField(
            model_name='clasificador',
            name='modelo_generado',
        ),
        migrations.AddField(
            model_name='caso',
            name='baseCaso',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='modelos.DataSet'),
        ),
        migrations.AddField(
            model_name='caso',
            name='claseE',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='clasificador',
            name='casosClasificar',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='modelos.DataSet'),
        ),
        migrations.AddField(
            model_name='clasificador',
            name='fechaHora',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='dataset',
            name='fecha',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='dataset',
            name='tipoBase',
            field=models.BooleanField(null=True),
        ),
        migrations.AlterField(
            model_name='atributo_entero',
            name='valor',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='atributo_float',
            name='valor',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='modelo',
            name='dataSet',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='modelos.DataSet'),
        ),
        migrations.AddField(
            model_name='clasificador',
            name='modeloUtilizado',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='modelos.Modelo'),
        ),
    ]
