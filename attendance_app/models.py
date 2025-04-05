from django.db import models
from django.contrib.auth.models import User
import os


def student_image_path(instance, filename):
    return f'student_images/{instance.admission_number}/{filename}'


class Course(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=20, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.code})"


class Unit(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='units')
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=20, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return f"{self.name} ({self.code}) - {self.course.name}"


class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    admission_number = models.CharField(max_length=20, unique=True)
    full_name = models.CharField(max_length=100)
    photo = models.ImageField(upload_to=student_image_path)
    course = models.ForeignKey(Course, on_delete=models.SET_NULL, null=True)
    registered_units = models.ManyToManyField(Unit, related_name='students')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.full_name} ({self.admission_number})"

    def delete(self, *args, **kwargs):
        # Delete the image file when the student is deleted
        if self.photo:
            if os.path.isfile(self.photo.path):
                os.remove(self.photo.path)
        super().delete(*args, **kwargs)


class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    unit = models.ForeignKey(Unit, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)
    is_present = models.BooleanField(default=True)

    class Meta:
        unique_together = ('student', 'unit', 'date')

    def __str__(self):
        return f"{self.student.full_name} - {self.unit.name} ({self.date})"