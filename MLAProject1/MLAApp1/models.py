from django.db import models

# Create your models here.
class ModelInfo(models.Model):
    nombre = models.CharField(max_length=50)
    algoritmo = models.CharField(max_length=20)
    dataSet = models.CharField(max_length=20)
    TiposDatos = models.CharField(max_length=50)

def __str__(self):
  return "nombre: {self.nombre}, algoritmo: {self.algoritmo}, dataset: {self.dataSet}"

