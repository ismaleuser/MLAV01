from django.shortcuts import render
from rest_pandas import PandasSimpleView
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
from sklearn import preprocessing
import csv
from django.http import HttpResponse, JsonResponse
import matplotlib.pyplot as plt

# Create your views here.

EJEMPLO = [146,184,235,200,226,251,299,273,281,304,203]


class Classifier:

    # Constructor de la clase
    def __init__(self, filename):

        # Leer el fichero csv
        f = open(filename)
        self.lines = f.readlines()
        f.close()
        # Leer la primera fila donde viene el tipo de dato de la columna
        self.format = self.lines[0].strip().split('\t')
        # Inicializar el arreglo de datos
        self.data = []
        # Recorrer las lineas del fichero a partir de la segunda
        for line in self.lines[1:]:
            # Leer cada línea
            self.fields = line.strip().split('\t')
            # inicializar listas de vectores con valores a ignorar, valores a utilizar y la clasificación
            self.ignore = []
            self.vector = []
            self.classification = []
            # Recorrer los elementos de la fila
            for i in range(len(self.fields)):
                # Llenar los vectores
                if self.format[i] == 'num':
                    self.vector.append(float(self.fields[i]))
                elif self.format[i] == 'comment':
                    self.ignore.append(self.fields[i])
                elif self.format[i] == 'class':
                    self.classification = self.fields[i]
            # Llenar la matriz con los datos
            self.data.append((self.classification, self.vector, self.ignore))
        # Convertir los datos en lista
        self.rawData = list(self.data)
        # get length of instance vector
        self.vlen = len(self.data[0][1])
        # Normalizar columna a columna
        for i in range(self.vlen):
            self.normalizeColumn(i)

    # Función de ejemplo para exportar fichero csv
    def guardar_dato(self):

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=normalizado.csv'

        # Creamos un escritor CSV usando a HttpResponse como "fichero"
        writer = csv.writer(response)
        writer.writerow(self.format)
        for i in range(len(self.fields)):
            writer.writerow(self.format)

        return response

    # Función de ejemplo para cargar un fichero csv
    def guardar_csv(request):
        # Creamos el objeto Httpresponse con la cabecera CSV apropiada.
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=ejemplo.csv'

        # Creamos un escritor CSV usando a HttpResponse como "fichero"
        writer = csv.writer(response)
        writer.writerow(['Year', 'Unruly Airline Passengers'])
        for (year, num) in zip(range(1995, 2006), EJEMPLO):
            writer.writerow([year, num])

        return response

    # Función para obtener la mediana de un vector
    def getMedian(self, alist):
        """return median of alist"""
        if alist == []:
            return []
        blist = sorted(alist)
        length = len(alist)
        if length % 2 == 1:
            # length of list is odd so return middle element
            return blist[int(((length + 1) / 2) - 1)]
        else:
            # length of list is even so compute midpoint
            v1 = blist[int(length / 2)]
            v2 = blist[(int(length / 2) - 1)]
            return (v1 + v2) / 2.0

    # Función para obtener la desviación estándar de un vector
    def getAbsoluteStandardDeviation(self, alist, median):
        """given alist and median return absolute standard deviation"""
        sum = 0
        for item in alist:
            sum += abs(item - median)
        return sum / len(alist)

    # Función para normalizar una columna de la matriz de datos
    def normalizeColumn(self, columnNumber):
        """given a column number, normalize that column in self.data"""
        # first extract values to list
        return JsonResponse({"Prueba2": columnNumber})
        col = [v[1][columnNumber] for v in self.data]
        median = self.getMedian(col)
        asd = self.getAbsoluteStandardDeviation(col, median)
        # print("Median: %f   ASD = %f" % (median, asd))
        self.medianAndDeviation.append((median, asd))
        for v in self.data:
            v[1][columnNumber] = (v[1][columnNumber] - median) / asd

    # Otra función creada específicamente para la SVM. Revisar código
    def normalizeColumnSVM(self, columnNumber, matriz):
        """given a column number, normalize that column in self.data"""
        # first extract values to list
        col = []
        for v in matriz:
            col.append(v[columnNumber])
        # verificar que sean números
        for i in col:
            if not i.isnumeric():
                i = 0
        median = self.getMedian(col)
        asd = self.getAbsoluteStandardDeviation(col, median)
        # print("Median: %f   ASD = %f" % (median, asd))
        self.medianAndDeviation.append((median, asd))
        for v in matriz:
            v[1][columnNumber] = (v[1][columnNumber] - median) / asd

    # Función para normalizar un vector
    def normalizeVector(self, v):
        """We have stored the median and asd for each column.
        We now use them to normalize vector v"""
        vector = list(v)
        for i in range(len(vector)):
            (median, asd) = self.medianAndDeviation[i]
            vector[i] = (vector[i] - median) / asd
        return vector

    # Función para calcular la distancia a utilizar en el KNN
    def manhattan(self, vector1, vector2):
        """Computes the Manhattan distance."""
        return sum(map(lambda v1, v2: abs(v1 - v2), vector1, vector2))

    # Función que calcula el vecino más cercano
    def nearestNeighbor(self, itemVector):
        """return nearest neighbor to itemVector"""
        return min([(self.manhattan(itemVector, item[1]), item)
                    for item in self.data])

    # Función que devuelve el valor de la clase del vecino más cercano
    def classify(self, itemVector):
        """Return class we think item Vector is in"""
        return (self.nearestNeighbor(self.normalizeVector(itemVector))[1][0])

    # Función que devuelve
    # def classifySVM(self, itemVector):
    #     """Return class we think item Vector is in"""
    #     return (self.svmClassify(self.normalizeVector(itemVector))[1][0])

    # Función que realiza el procesamiento total de la SVM
    def svmModel(self, training_filename, test_filename):
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
        Scaled = preprocessing.scale(matriz)
        Normalized = preprocessing.normalize(matriz, norm='l2')
        robustScaled = preprocessing.robust_scale(matriz, axis=0, with_centering=True, with_scaling=True,
                                                  quantile_range=(25.0, 75.0), copy=True)
#        return JsonResponse({"Prueba": list(robustScaled)})



        # Se crea el modelo a partir del clasificador seleccionado y con los datos escogidos
        clf = svm.SVC(gamma=0.5, C=100.)
        clf.fit(matriz, classify)



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

        # Devuelvo en formato JSON los resultados
#        resultado.append("%4.2f%% correct" % (numCorrect))
        return JsonResponse({"Clasificado": list(res),
                             "Original": classify,
                             "Correctamente clasificados": numCorrect,
                             "Mal clasificados": numIncorrect})

    def prueba(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import datasets, svm

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X = X[y != 0, :2]
        y = y[y != 0]
        n_sample = len(X)
        np.random.seed(0)
        order = np.random.permutation(n_sample)
        X = X[order]
        y = y[order].astype(np.float)
        X_train = X[:int(.9 * n_sample)]
        y_train = y[:int(.9 * n_sample)]
        X_test = X[int(.9 * n_sample):]
        y_test = y[int(.9 * n_sample):]
        for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')): clf = svm.SVC(kernel=kernel, gamma=10)
        clf.fit(X_train, y_train)
        plt.figure(fig_num)
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)
        # Circle out the test data plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10, edgecolor='k')
        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        # Put the result into a color plot Z = Z.reshape(XX.shape) plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired) plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
        plt.title(kernel)
        plt.savefig('image2.png')
        image_data = open("image2.png", "rb").read()
        return HttpResponse(image_data, content_type="image/png")

    def prueba2(self):

        import datetime
        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        # Create the PdfPages object to which we will save the pages:
        # The with statement makes sure that the PdfPages object is closed properly at
        # the end of the block, even if an Exception occurs.
        with PdfPages('multipage_pdf.pdf') as pdf:
            plt.figure(figsize=(3, 3))
            plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
            plt.title('Page One')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            plt.rc('text', usetex=True)
            plt.figure(figsize=(8, 6))
            x = np.arange(0, 5, 0.1)
            plt.plot(x, np.sin(x), 'b-')
            plt.title('Page Two')
            pdf.savefig()
            plt.close()

            plt.rc('text', usetex=False)
            fig = plt.figure(figsize=(4, 5))
            plt.plot(x, x * x, 'ko')
            plt.title('Page Three')
            pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
            plt.close()

            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Multipage PDF Example'
            d['Author'] = u'Jouni K. Sepp\xe4nen'
            d['Subject'] = 'How to create a multipage pdf file and set its metadata'
            d['Keywords'] = 'PdfPages multipage keywords author title subject'
            d['CreationDate'] = datetime.datetime(2009, 11, 13)
            d['ModDate'] = datetime.datetime.today()

        return HttpResponse(pdf, content_type="text/pdf")





# Función para ejecutar el algoritmo KNN
def knn(training_filename, test_filename):
    """Función para clasificar con KNN"""
    # Se crea el objeto de la clase
    classifier = Classifier(training_filename)
    # Se accede al fichero de prueba
    f = open(test_filename)
    lines = f.readlines()
    f.close()
    # Se inicializan las variables necesarias
    resultado = []
    numCorrect = 0.0
    # Se recorren las líneas del fichero
    for line in lines:
        data = line.strip().split('\t')
        vector = []
        classInColumn = -1
        for i in range(len(classifier.format)):
            if classifier.format[i] == 'num':
                vector.append(float(data[i]))
            elif classifier.format[i] == 'class':
                classInColumn = i
        theClass = classifier.classify(vector)
        prefix = '-'
        if theClass == data[classInColumn]:
            # it is correct
            numCorrect += 1
            prefix = '+'
        resultado.append("%s  %12s  %s" % (prefix, theClass, line))
    resultado.append("%4.2f%% correct" % (numCorrect * 100 / len(lines)))
    return JsonResponse({"Clasificación": resultado})
    # print("%4.2f%% correct" % (numCorrect * 100 / len(lines)))

# Función que llama a la función que ejecuta el algoritmo KNN.
# Esta es la que se llama desde urls.py
def llamadaknn(request):
    # ejemplo = Classifier('DsDesercion2.csv')
    return knn('data/DsDesercionTraining.csv', 'data/DsDesercionTest.csv')
    # return ejemplo.guardar_dato()

# Función que crea el objeto de la clase y ejecuta la función para aplicar la SVM
# Esta es la función que se llama desde urls.py
def llamadasvm(request):
    ejemplo = Classifier('data/Training.csv')
    return ejemplo.svmModel('data/Training.csv', 'data/Test.csv')

def llamadaPrueba(request):
    ejemplo = Classifier('data/Training.csv')
    return ejemplo.prueba()

def llamadaPrueba2(request):
    ejemplo = Classifier('data/Training.csv')
    return ejemplo.prueba2()

    # return ejemplo.guardar_dato()

# Función de prueba que muestra una imagen como respuesta de la función
def my_image(request):
    image_data = open("data/Image.png", "rb").read()
    return HttpResponse(image_data, content_type="image/png")

# Función de prueba para leer un fichero csv
class TimeSeriesView(PandasSimpleView):
    def get_data(self, request):
        return pd.read_csv('data/DsDesercion1.csv')

