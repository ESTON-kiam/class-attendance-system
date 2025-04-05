from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Course, Unit, Student
import os
import base64
from django.conf import settings


class CourseForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = ['name', 'code', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }
        labels = {
            'name': 'Course Name',
            'code': 'Course Code',
            'description': 'Description'
        }


class UnitForm(forms.ModelForm):
    class Meta:
        model = Unit
        fields = ['course', 'name', 'code', 'description']
        labels = {
            'course': 'Associated Course',
            'name': 'Unit Name',
            'code': 'Unit Code',
            'description': 'Description'
        }


class StudentRegistrationForm(forms.ModelForm):
    username = forms.CharField(max_length=150, required=True, label='Username')
    password = forms.CharField(widget=forms.PasswordInput, required=True, label='Password')
    email = forms.EmailField(required=True, label='Email Address')
    photo_data = forms.CharField(widget=forms.HiddenInput(), required=False)
    gender = forms.ChoiceField(choices=Student.GENDER_CHOICES, widget=forms.RadioSelect, label='Gender')

    class Meta:
        model = Student
        fields = ['admission_number', 'full_name', 'course', 'registered_units', 'gender', 'phone_number']
        widgets = {
            'registered_units': forms.CheckboxSelectMultiple,
            'phone_number': forms.TextInput(attrs={'placeholder': 'e.g. +254712345678'}),
        }
        labels = {
            'admission_number': 'Admission Number',
            'full_name': 'Full Name',
            'course': 'Enrolled Course',
            'registered_units': 'Select Units',
            'phone_number': 'Phone Number'
        }

    def clean_username(self):
        username = self.cleaned_data['username']
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("This username is already taken. Please choose another one.")
        return username

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

    def clean_admission_number(self):
        admission_number = self.cleaned_data['admission_number']
        if Student.objects.filter(admission_number=admission_number).exists():
            raise forms.ValidationError("This admission number is already registered.")
        return admission_number

    def save(self, commit=True):
        user = User.objects.create_user(
            username=self.cleaned_data['username'],
            email=self.cleaned_data['email'],
            password=self.cleaned_data['password']
        )

        student = super().save(commit=False)
        student.user = user
        student.email = self.cleaned_data['email']

        photo_data = self.cleaned_data.get('photo_data')
        if photo_data:
            student.photo = self.save_photo_from_data(photo_data, student.admission_number)

        if commit:
            student.save()
            self.save_m2m()

        return student

    def save_photo_from_data(self, photo_data, admission_number):
        format, imgstr = photo_data.split(';base64,')
        ext = format.split('/')[-1]
        filename = f"{admission_number}_profile.{ext}"
        filepath = os.path.join(settings.MEDIA_ROOT, 'student_photos', admission_number, filename)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(imgstr))

        return f'student_photos/{admission_number}/{filename}'


class StudentUpdateForm(forms.ModelForm):
    photo_data = forms.CharField(widget=forms.HiddenInput(), required=False)
    gender = forms.ChoiceField(choices=Student.GENDER_CHOICES, widget=forms.RadioSelect, label='Gender')

    class Meta:
        model = Student
        fields = ['full_name', 'photo', 'course', 'registered_units', 'gender', 'phone_number']
        widgets = {
            'photo': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'capture': 'environment'  # For mobile devices to use camera directly
            }),
            'registered_units': forms.CheckboxSelectMultiple,
            'phone_number': forms.TextInput(attrs={'placeholder': 'e.g. +254712345678'}),
        }
        labels = {
            'full_name': 'Full Name',
            'photo': 'Profile Photo',
            'course': 'Enrolled Course',
            'registered_units': 'Registered Units',
            'phone_number': 'Phone Number'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['photo'].required = False
        if self.instance and self.instance.photo:
            self.fields['photo'].widget.attrs['data-current-image'] = self.instance.photo.url

    def save(self, commit=True):
        student = super().save(commit=False)

        # Handle webcam photo capture
        photo_data = self.cleaned_data.get('photo_data')
        if photo_data:
            student.photo = self.save_photo_from_data(photo_data, student.admission_number)

        # Handle photo removal
        if self.cleaned_data.get('photo-clear'):
            student.photo.delete(save=False)
            student.photo = None

        if commit:
            student.save()
            self.save_m2m()

        return student

    def save_photo_from_data(self, photo_data, admission_number):
        format, imgstr = photo_data.split(';base64,')
        ext = format.split('/')[-1]
        filename = f"{admission_number}_profile_updated.{ext}"
        filepath = os.path.join(settings.MEDIA_ROOT, 'student_photos', admission_number, filename)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(imgstr))

        return f'student_photos/{admission_number}/{filename}'