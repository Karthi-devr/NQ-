from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('performance/', views.performance, name='performance'),
    path('training-data/', views.training_data, name='training_data'),
    path('save-prediction/', views.save_prediction, name='save_prediction'),
    path('update-outcome/<int:day_id>/', views.update_outcome, name='update_outcome'),
    path('retrain/', views.retrain_model, name='retrain_model'),
]
