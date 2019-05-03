from django.db import models


class Fichero(models.Model):

    ACLASS = 'clase'
    AATR = 'atributo'
    AALG = 'algoritmo'

    VARIABLES = (
        (ACLASS, 'clase'),
        (AATR, 'atributo'),
        (AALG, 'algoritmo'),
    )

    nombre = models.CharField(max_length=100)
    atributos = models.CharField(max_length=10,choices=VARIABLES)

