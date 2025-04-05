from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views

from attendance_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.admin_dashboard, name='home'),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='attendance_app/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('attendance/', include('attendance_app.urls')),
]