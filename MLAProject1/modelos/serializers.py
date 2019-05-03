from rest_framework import serializers
from .models import Atributo_Entero, Atributo_String, Atributo_Float, Clasificador, Caso, DataSet, UploadFile, Algoritmo, Modelo

class Algoritmo_serializer(serializers.ModelSerializer):
    class Meta:
        model = Algoritmo
        fields =('nombre', 'info')

class DataSet_serializer(serializers.ModelSerializer):
    class Meta:
        model = DataSet
        fields = ('nombre', 'fecha', 'tipoBase')

class Caso_serializer(serializers.ModelSerializer):
    class Meta:
        model = Caso
        fields = ('nombre', 'baseCaso', 'claseS', 'claseE')

class Atributo_String_serializer(serializers.ModelSerializer):
    class Meta:
        model = Atributo_String
        fields = ('nombre', 'valor', 'dominio', 'caso')

class Atributo_Entero_serializer(serializers.ModelSerializer):
    class Meta:
        model = Atributo_Entero
        fields = ('nombre', 'valor', 'caso')

class Atributo_Float_serializer(serializers.ModelSerializer):
    class Meta:
        model = Atributo_Float
        fields = ('nombre', 'valor', 'caso')

class Modelo_seralizer(serializers.ModelSerializer):
    class Meta:
        model = Modelo
        fields = ('nombre', 'algoritmo', 'direccion', 'dataSet', 'fechaGenerado')

class Clasificador_serializer(serializers.ModelSerializer):
    class Meta:
        model = Clasificador
        fields = ('base_caso', 'modeloUtilizado', 'fechaHora')


class UploadFile_serializer(serializers.ModelSerializer):

    class Meta:
        model = UploadFile
        read_only_fields = ('creation_date', 'dataset', 'file')

