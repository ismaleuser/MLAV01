from django.db import models

# Create your models here.

class DataSet(models.Model):

    identif = models.IntegerField()
    nombre = models.CharField(max_length=20)
    cantAtributos = models.IntegerField()
