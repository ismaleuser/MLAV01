from django.contrib import admin

# Register your models here.
from.models import ModelInfo
from.AppFichero import Fichero
admin.site.register(ModelInfo)
admin.site.register(Fichero)
