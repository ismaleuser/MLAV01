from django.conf.urls import url
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.caso_list),
    path("SVM/", views.svmModel),
    path("DT/", views.dtcModel),
    path("Load/", views.upload_file),
]
