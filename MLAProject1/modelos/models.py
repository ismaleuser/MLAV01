from django.db import models

# Create your models here.

class Algoritmo(models.Model):
    nombre = models.CharField(max_length=50)
    info = models.CharField(max_length=250)

    def __str__(self):
        return self.nombre

class DataSet(models.Model):
    nombre = models.CharField(max_length= 50)
    fecha = models.DateTimeField(null=True, blank=True)
    tipoBase = models.BooleanField(null=True)

    def __str__(self):
        return self.nombre

class Caso(models.Model):
    nombre = models.CharField(max_length= 50)
    baseCaso = models.ForeignKey(DataSet, on_delete=models.SET_NULL, null=True)
    claseS = models.CharField(max_length= 50)
    claseE = models.IntegerField(null=True)

    def __str__(self):
        return self.nombre

class Atributo_String(models.Model):
    nombre = models.CharField(max_length= 50)
    valor = models.CharField(max_length=50)
    dominio = models.TextField(max_length=255)
    caso = models.ForeignKey(Caso, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.nombre

class Atributo_Entero(models.Model):
    nombre = models.CharField(max_length= 50)
    valor = models.IntegerField(null=True)
    caso = models.ForeignKey(Caso, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.nombre

class Atributo_Float(models.Model):
    nombre = models.CharField(max_length= 50)
    valor = models.FloatField(null=True)
    caso = models.ForeignKey(Caso, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.nombre

class Modelo(models.Model):
    algoritmo = models.ForeignKey(Algoritmo, on_delete=models.SET_NULL, null=True)
    nombre = models.CharField(max_length= 50)
    direccion = models.CharField(max_length= 250)
    dataSet = models.ForeignKey(DataSet, on_delete=models.SET_NULL, null=True)
    fechaGenerado = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.nombre

class Clasificador(models.Model):
    casosClasificar = models.ForeignKey(DataSet, on_delete=models.SET_NULL, null=True)
    modeloUtilizado = models.ForeignKey(Modelo, on_delete=models.SET_NULL, null=True)
    fechaHora = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.fecha.value_to_string()


class UploadFile(models.Model):
    file = models.FileField(upload_to='datasets/', null=True)
    creation_date = models.DateTimeField(auto_now_add=True, null=True)
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE, null=True)

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        self.file.name = str('dataset.csv')
        super().save(force_insert, force_update, using, update_fields)

class UpLoadTestModel(models.Model):
    titulo = models.CharField(max_length=100)
    texto = models.TextField(null=True, blank=True)
    archivo = models.FileField(upload_to="uploads/", null=True, blank=True)