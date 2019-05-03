from django.shortcuts import render
from rest_pandas import PandasSimpleView
import pandas as pd

# Create your views here.
import csv
from django.http import HttpResponse, JsonResponse

EJEMPLO = [146,184,235,200,226,251,299,273,281,304,203]


class Classifier:

    def __init__(self, filename):

        f = open(filename)
        self.lines = f.readlines()
        f.close()
        #
        self.format = self.lines[0].strip().split('\t')
        self.data = []
        for line in self.lines[1:]:
            self.fields = line.strip().split('\t')
            self.ignore = []
            self.vector = []
            self.classification = []
            for i in range(len(self.fields)):
                if self.format[i] == 'num':
                    self.vector.append(float(self.fields[i]))
                elif self.format[i] == 'comment':
                    self.ignore.append(self.fields[i])
                elif self.format[i] == 'class':
                    self.classification = self.fields[i]
            self.data.append((self.classification, self.vector, self.ignore))
        self.rawData = list(self.data)
        # get length of instance vector
        self.vlen = len(self.data[0][1])
        # Normalizar columna a columna
        for i in range(self.vlen):
            self.normalizeColumn(i)

    def guardar_dato(self):

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=normalizado.csv'

        # Creamos un escritor CSV usando a HttpResponse como "fichero"
        writer = csv.writer(response)
        writer.writerow(self.format)
        for i in range(len(self.fields)):
            writer.writerow(self.format)

        return response


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


    def getAbsoluteStandardDeviation(self, alist, median):
        """given alist and median return absolute standard deviation"""
        sum = 0
        for item in alist:
            sum += abs(item - median)
        return sum / len(alist)


    def normalizeColumn(self, columnNumber):
        """given a column number, normalize that column in self.data"""
        # first extract values to list
        col = [v[1][columnNumber] for v in self.data]
        median = self.getMedian(col)
        asd = self.getAbsoluteStandardDeviation(col, median)
        # print("Median: %f   ASD = %f" % (median, asd))
        self.medianAndDeviation.append((median, asd))
        for v in self.data:
            v[1][columnNumber] = (v[1][columnNumber] - median) / asd


    def normalizeVector(self, v):
        """We have stored the median and asd for each column.
        We now use them to normalize vector v"""
        vector = list(v)
        for i in range(len(vector)):
            (median, asd) = self.medianAndDeviation[i]
            vector[i] = (vector[i] - median) / asd
        return vector

    def manhattan(self, vector1, vector2):
        """Computes the Manhattan distance."""
        return sum(map(lambda v1, v2: abs(v1 - v2), vector1, vector2))

    def nearestNeighbor(self, itemVector):
        """return nearest neighbor to itemVector"""
        return min([(self.manhattan(itemVector, item[1]), item)
                    for item in self.data])

    def classify(self, itemVector):
        """Return class we think item Vector is in"""
        return (self.nearestNeighbor(self.normalizeVector(itemVector))[1][0])

def knn(training_filename, test_filename):
    """Función para clasificar con KNN"""
    classifier = Classifier(training_filename)
    f = open(test_filename)
    lines = f.readlines()
    f.close()
    numCorrect = 0.0
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
       # print("%s  %12s  %s" % (prefix, theClass, line))
    return JsonResponse({"prueba": "%4.2f%% correct" % (numCorrect * 100 / len(lines))})
    # print("%4.2f%% correct" % (numCorrect * 100 / len(lines)))

def llamada(request):
    ejemplo = Classifier('DsDesercion2.csv')
    return knn('DsDesercion2.csv', 'DsDesercion2C.csv')
    return ejemplo.guardar_dato()

def my_image(request):
    image_data = open("data/Image.png", "rb").read()
    return HttpResponse(image_data, content_type="image/png")


class TimeSeriesView(PandasSimpleView):
    def get_data(self, request):
        return pd.read_csv('data/DsDesercion1.csv')
