from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.
from .models import ModelInfo

def getModels(request):
    models = ModelInfo.objects.all()
    data = list(models.values("nombre", "algoritmo", "dataSet"))
    return JsonResponse(data, safe = False)

def getModelDetail(request, param):
    data = ModelInfo.objects.filter(nombre=param)
    return JsonResponse(list(data.values("nombre", "algoritmo", "dataSet")),safe=False)
