import csv
import numpy as np
from django.views import generic
from django.shortcuts import render
from django.http import HttpResponse

from rest_framework import viewsets

from .models import DataSet
from .serializers import DataSetSerializer

# Create your views here.
class DataSetView(viewsets.ModelViewSet):
    queryset = DataSet.objects.all()
    serializer_class = DataSetSerializer
#
# class CSVTestView(generic.FormView):
#     tabla = np.random.random((3,3))
#     encabezado = ('num', 'text', 'class')
#
#     with open('tabla.csv','w',newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(encabezado)
#         writer.writerows(tabla)

# class DataSetView(generic.FormView):

UNRULY_PASSENGERS = [146,184,235,200,226,251,299,273,281,304,203]

def unruly_passengers_csv(request):
    # Creamos el objeto Httpresponse con la cabecera CSV apropiada.
    response = HttpResponse(mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=unruly.csv'

    # Creamos un escritor CSV usando a HttpResponse como "fichero"
    writer = csv.writer(response)
    writer.writerow(['Year', 'Unruly Airline Passengers'])
    for (year, num) in zip(range(1995, 2006), UNRULY_PASSENGERS):
        writer.writerow([year, num])
    return response