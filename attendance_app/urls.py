from django.urls import path
from . import views

urlpatterns = [
    # Admin URLs
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/courses/', views.course_list, name='course_list'),
    path('admin/courses/add/', views.add_course, name='add_course'),
    path('admin/units/', views.unit_list, name='unit_list'),
    path('admin/units/add/', views.add_unit, name='add_unit'),
    path('admin/students/', views.student_list, name='student_list'),
    path('admin/students/register/', views.register_student, name='register_student'),
    path('admin/students/edit/<int:pk>/', views.edit_student, name='edit_student'),
    path('admin/students/delete/<int:pk>/', views.delete_student, name='delete_student'),
    path('admin/attendance/', views.attendance_records, name='attendance_records'),

    # Student URLs
    path('student/dashboard/', views.student_dashboard, name='student_dashboard'),
    path('student/mark-attendance/', views.mark_attendance, name='mark_attendance'),
    path('student/attendance-records/', views.student_attendance_records, name='student_attendance_records'),
]