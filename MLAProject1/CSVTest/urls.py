from django.urls import path
from . import views

urlpatterns = (
    path("", views.Classifier.guardar_csv),
    path("Image/", views.my_image),
    path("CSV/", views.TimeSeriesView.get_data),
    path("KNN/", views.llamadaknn),
    path("SVM/", views.llamadasvm),
    path("Prueba/", views.llamadaPrueba2)
)