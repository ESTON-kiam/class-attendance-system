from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Course, Unit, Student


class CourseForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = ['name', 'code', 'description']


class UnitForm(forms.ModelForm):
    class Meta:
        model = Unit
        fields = ['course', 'name', 'code', 'description']


class StudentRegistrationForm(forms.ModelForm):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)
    email = forms.EmailField()

    class Meta:
        model = Student
        fields = ['admission_number', 'full_name', 'photo', 'course', 'registered_units']

    def save(self, commit=True):
        # Create the User first
        user = User.objects.create_user(
            username=self.cleaned_data['username'],
            password=self.cleaned_data['password'],
            email=self.cleaned_data['email']
        )

        # Then create the Student
        student = super().save(commit=False)
        student.user = user
        if commit:
            student.save()
            self.save_m2m()
        return student


class StudentUpdateForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ['full_name', 'photo', 'course', 'registered_units']