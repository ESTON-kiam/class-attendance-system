from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import login, authenticate
from django.http import JsonResponse, HttpResponse
from django.conf import settings

from .face_recognition import FaceRecognition
from .models import Course, Unit, Student, Attendance
from .forms import CourseForm, UnitForm, StudentRegistrationForm, StudentUpdateForm

import cv2
import os
from datetime import datetime


def is_admin(user):
    return user.is_superuser


# Admin Views
@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    courses = Course.objects.all().count()
    units = Unit.objects.all().count()
    students = Student.objects.all().count()
    context = {
        'courses': courses,
        'units': units,
        'students': students,
    }
    return render(request, 'attendance_app/admin/dashboard.html', context)


@login_required
@user_passes_test(is_admin)
def course_list(request):
    courses = Course.objects.all()
    return render(request, 'attendance_app/admin/course_list.html', {'courses': courses})


@login_required
@user_passes_test(is_admin)
def add_course(request):
    if request.method == 'POST':
        form = CourseForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('course_list')
    else:
        form = CourseForm()
    return render(request, 'attendance_app/admin/add_course.html', {'form': form})


@login_required
@user_passes_test(is_admin)
def unit_list(request):
    units = Unit.objects.all()
    return render(request, 'attendance_app/admin/unit_list.html', {'units': units})


@login_required
@user_passes_test(is_admin)
def add_unit(request):
    if request.method == 'POST':
        form = UnitForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('unit_list')
    else:
        form = UnitForm()
    return render(request, 'attendance_app/admin/add_unit.html', {'form': form})


@login_required
@user_passes_test(is_admin)
def student_list(request):
    students = Student.objects.all()
    return render(request, 'attendance_app/admin/student_list.html', {'students': students})


@login_required
@user_passes_test(lambda u: u.is_superuser)
def register_student(request):
    if request.method == 'POST':
        form = StudentRegistrationForm(request.POST)
        if form.is_valid():
            try:
                form.save()
                messages.success(request, 'Student registered successfully!')
                return redirect('student_list')
            except IntegrityError as e:
                messages.error(request, f'Error saving student: {str(e)}')
        else:
            # Collect all form errors
            error_messages = []
            for field, errors in form.errors.items():
                for error in errors:
                    error_messages.append(f"{field}: {error}")
            messages.error(request, 'Please correct the errors below.')
            messages.error(request, ' '.join(error_messages))
    else:
        form = StudentRegistrationForm()

    return render(request, 'attendance_app/admin/register_student.html', {
        'form': form,
        'form_errors': form.errors if request.method == 'POST' else None
    })
@login_required
@user_passes_test(is_admin)
def edit_student(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        form = StudentUpdateForm(request.POST, request.FILES, instance=student)
        if form.is_valid():
            form.save()
            return redirect('student_list')
    else:
        form = StudentUpdateForm(instance=student)
    return render(request, 'attendance_app/admin/edit_student.html', {'form': form, 'student': student})


@login_required
@user_passes_test(is_admin)
def delete_student(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        student.delete()
        return redirect('student_list')
    return render(request, 'attendance_app/admin/delete_student.html', {'student': student})


@login_required
@user_passes_test(is_admin)
def attendance_records(request):
    attendances = Attendance.objects.all().order_by('-date', '-time')
    return render(request, 'attendance_app/admin/attendance_records.html', {'attendances': attendances})


# Student Views
@login_required
def student_dashboard(request):
    if not hasattr(request.user, 'student'):
        return HttpResponse("You are not registered as a student.", status=403)

    student = request.user.student
    today = datetime.now().date()

    # Get today's attendance
    today_attendance = Attendance.objects.filter(student=student, date=today)

    # Get all registered units
    registered_units = student.registered_units.all()

    context = {
        'student': student,
        'today_attendance': today_attendance,
        'registered_units': registered_units,
    }
    return render(request, 'attendance_app/student/dashboard.html', context)


@login_required
def mark_attendance(request):
    if not hasattr(request.user, 'student'):
        return JsonResponse({'success': False, 'message': 'You are not registered as a student.'})

    student = request.user.student

    # Initialize face recognition
    face_recognizer = FaceRecognition()
    face_recognizer.load_student_images()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    recognized = False
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts and not recognized:
        ret, frame = cap.read()
        if not ret:
            break

        # Recognize faces in the frame
        recognized_ids = face_recognizer.recognize_face(frame)

        if student.id in recognized_ids:
            recognized = True
            break

        attempts += 1

    cap.release()

    if recognized:
        # Mark attendance for all registered units
        today = datetime.now().date()
        for unit in student.registered_units.all():
            Attendance.objects.get_or_create(
                student=student,
                unit=unit,
                date=today,
                defaults={'is_present': True}
            )

        return JsonResponse({'success': True, 'message': 'Attendance marked successfully!'})
    else:
        return JsonResponse(
            {'success': False, 'message': 'Face recognition failed. Please try again or contact admin.'})


@login_required
def student_attendance_records(request):
    if not hasattr(request.user, 'student'):
        return HttpResponse("You are not registered as a student.", status=403)

    student = request.user.student
    attendances = Attendance.objects.filter(student=student).order_by('-date', '-time')

    return render(request, 'attendance_app/student/attendance_records.html', {'attendances': attendances})


@login_required
@user_passes_test(is_admin)
def edit_course(request, pk):
    course = get_object_or_404(Course, pk=pk)
    if request.method == 'POST':
        form = CourseForm(request.POST, instance=course)
        if form.is_valid():
            form.save()
            return redirect('course_list')
    else:
        form = CourseForm(instance=course)
    return render(request, 'attendance_app/admin/edit_course.html', {'form': form, 'course': course})


@login_required
@user_passes_test(is_admin)
def delete_course(request, pk):
    course = get_object_or_404(Course, pk=pk)
    if request.method == 'POST':
        course.delete()
        return redirect('course_list')
    return render(request, 'attendance_app/admin/delete_course.html', {'course': course})