from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from .models import Atributo_Entero, Clasificador, Caso, UploadFile, DataSet
from .serializers import Atributo_String_serializer, Clasificador_serializer, Caso_serializer, UploadFile_serializer, DataSet_serializer
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
import csv
from sklearn import tree
from rest_framework import viewsets
import graphviz


# Create your views here.

class JSONResponse(HttpResponse):
    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content, **kwargs)


def guardar_casos(request):
    clasificador1 = Clasificador(base_caso="Base de ejemplo", modelo_generado="Modelo")
    clasificador1.save()
    caso = Caso(nombre="Caso1", clase="Una", clasificador=clasificador1)
    caso.save()

    clasificador2 = Clasificador(base_caso="Base de ejemplo", modelo_generado="Modelo2")
    clasificador2.save()
    caso = Caso(nombre="Caso2", clase="Dos", clasificador=clasificador2)
    caso.save()


@csrf_exempt
def caso_list(request):
    if request.method == 'GET':
        guardar_casos(request)
        casos = Caso.objects.all()
        serializer = Caso_serializer(casos, many=True)
        return JSONResponse(serializer.data)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = Caso_serializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JSONResponse(serializer.data, status=201)
        return JSONResponse(serializer.errors, status=400)


def svmModel(request, test_filename = 'data/Training.csv'):
    # Cargar los datos del fichero
    f = open(test_filename)
    lines = f.readlines()
    f.close()
    # Inicializar los vectores de datos
    matriz = []
    classify = []
    # Convierto en una lista los datos de la primera fila que vienen separados por coma
    format = list(lines[0].strip().split(','))
    # Recorro las líneas del fichero a partir de la segunda
    for line in lines[1:]:
        # Inicializo variable temporal
        vector = []
        # Convierto en lista los elementos de la fila que vienen separados por coma
        lista = list(line.split(','))
        # Recorro la lista
        for i in range(len(lista)):
            # Lleno los vectores correspondientes
            if format[i] == 'num':
                vector.append(float(lista[i]))
            elif format[i] == 'class':
                classify.append(list(lista[i][:-2]))
        # Se llena la matriz
        matriz.append(vector)

    # Se aplica el preprocesamiento
    # Scaled = preprocessing.scale(matriz)
    # Normalized = preprocessing.normalize(matriz, norm='l2')
    # robustScaled = preprocessing.robust_scale(matriz, axis=0, with_centering=True, with_scaling=True,
    #                                          quantile_range=(25.0, 75.0), copy=True)
    # return JSONResponse({"Prueba": list(robustScaled)})

    # Se crea el modelo a partir el clasificador seleccionado y con los datos escogidos
    clf = svm.SVC(gamma=0.1, C=100.)
    clf.fit(matriz, classify)

    # lista = clf.n_support_

    # return JSONResponse(lista)

    # Se guarda el modelo en el fichero
    joblib.dump(clf, 'modelo.joblib')

    # Para clasificar los nuevos datos
    # Cargar el modelo guardado
    clf2 = joblib.load('modelo.joblib')

    # Inicializar variables para ver resultados
    numCorrect = 0.0
    numIncorrect = 0.0

    # Aplicar clasificación a los datos
    res = clf2.predict(matriz)
    # Se inicaliza una lista para comparar resultados
    Solu = []
    # Convertir el resultado en una lista
    resList = list(res)
    # Recorrer la lista para adicionar a una lista cada elemento del resultado convertido en lista
    for j in range(len(resList)):
        Solu.append(list(res[j]))

    # Comparar resultados obtenidos con los utilizados en la creación del modelo
    resultado = []
    for i in range(len(resList)):
        if Solu[i] == classify[i]:
            numCorrect += 1
        else:
            numIncorrect += 1

    for i in range(len(matriz)):
        matriz[i].append(Solu[i])

    with open('resultado.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(format)
        writer.writerows(matriz)

    # Devuelvo en formato JSON los resultados
    #        resultado.append("%4.2f%% correct" % (numCorrect))
    return JSONResponse({"Clasificado": Solu,
                         "Original": classify,
                         "Correctamente clasificados": numCorrect,
                         "Mal clasificados": numIncorrect})

from django.http import HttpResponseRedirect
from django.shortcuts import render
#from .forms import UploadFileForm
from django.contrib import messages
#from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect

from .forms import FormEntrada
from .models import UpLoadTestModel

def upLoadTest(request):
    form = FormEntrada(request.POST, request.FILES)
    if form.is_valid():
        titulo = request.POST['titulo']
        texto = request.POST['texto']
        archivo = request.FILES['archivo']

        insert = UpLoadTestModel(titulo=titulo, texto=texto, archivo=archivo)
        insert.save()

                # return HttpResponseRedirect(reverse('index'))
    else:
        messages.error(request, "Error al procesar el formulario")
    return JSONResponse({"Éxito":"si"})

def dtcModel(request, test_filename = 'data/Training.csv'):
    # Cargar los datos del fichero
    f = open(test_filename)
    lines = f.readlines()
    f.close()
    # Inicializar los vectores de datos
    matriz = []
    classify = []
    # Convierto en una lista los datos de la primera fila que vienen separados por coma
    format = list(lines[0].strip().split(','))
    # Recorro las líneas del fichero a partir de la segunda
    for line in lines[1:]:
        # Inicializo variable temporal
        vector = []
        # Convierto en lista los elementos de la fila que vienen separados por coma
        lista = list(line.split(','))
        # Recorro la lista
        for i in range(len(lista)):
            # Lleno los vectores correspondientes
            if format[i] == 'num':
                vector.append(float(lista[i]))
            elif format[i] == 'class':
                classify.append(list(lista[i][:-2]))
        # Se llena la matriz
        matriz.append(vector)

    # Se aplica el preprocesamiento
    # Scaled = preprocessing.scale(matriz)
    # Normalized = preprocessing.normalize(matriz, norm='l2')
    # robustScaled = preprocessing.robust_scale(matriz, axis=0, with_centering=True, with_scaling=True,
    #                                          quantile_range=(25.0, 75.0), copy=True)
    # return JSONResponse({"Prueba": list(robustScaled)})

    # Se crea el modelo a partir el clasificador seleccionado y con los datos escogidos
    clf2 = tree.DecisionTreeClassifier()
    clf2.fit(matriz, classify)


    # lista = clf.n_support_

    # return JSONResponse(lista)

    # Se guarda el modelo en el fichero
    joblib.dump(clf2, 'modeloarbol.joblib')

    # Para clasificar los nuevos datos
    # Cargar el modelo guardado
    clf3 = joblib.load('modeloarbol.joblib')

    # Inicializar variables para ver resultados
    numCorrect = 0.0
    numIncorrect = 0.0

    # Aplicar clasificación a los datos
    res = clf3.predict(matriz)
    prob = clf3.predict_proba(matriz)
    listProb = list(prob)

    # dot_data = tree.export_graphviz(clf2, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("Prueba")


    # Se inicaliza una lista para comparar resultados
    Solu = []
    # Convertir el resultado en una lista
    resList = list(res)
    # Recorrer la lista para adicionar a una lista cada elemento del resultado convertido en lista
    for j in range(len(resList)):
        Solu.append(list(res[j]))

    # Comparar resultados obtenidos con los utilizados en la creación del modelo
    resultado = []
    for i in range(len(resList)):
        if Solu[i] == classify[i]:
            numCorrect += 1
        else:
            numIncorrect += 1

    for i in range(len(matriz)):
        matriz[i].append(Solu[i])

    with open('resultado.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(format)
        writer.writerows(matriz)

    # Devuelvo en formato JSON los resultados
    #        resultado.append("%4.2f%% correct" % (numCorrect))
    return JSONResponse({"Clasificado": Solu,
                         "Original": classify,
                         "Correctamente clasificados": numCorrect,
                         "Mal clasificados": numIncorrect,
                         "Probabilidad": listProb,
                         })

from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm

def handle_uploaded_file(f):
    with open('DataSet.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return JSONResponse({"Entró": "Aquí"})

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('Base.html')
    else:
        form = UploadFileForm()
    return render(request, 'import.html', {'form': form})
