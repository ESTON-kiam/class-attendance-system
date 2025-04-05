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


class UnitForm(forms.ModelForm):
    class Meta:
        model = Unit
        fields = ['course', 'name', 'code', 'description']

class StudentRegistrationForm(forms.ModelForm):
    username = forms.CharField(max_length=150, required=True)
    password = forms.CharField(widget=forms.PasswordInput, required=True)
    email = forms.EmailField(required=True)
    photo_data = forms.CharField(widget=forms.HiddenInput(), required=False)

    class Meta:
        model = Student
        fields = ['admission_number', 'full_name', 'course', 'registered_units', 'gender', 'phone_number']
        widgets = {
            'registered_units': forms.CheckboxSelectMultiple,
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
        # Create the User first
        user = User.objects.create_user(
            username=self.cleaned_data['username'],
            email=self.cleaned_data['email'],
            password=self.cleaned_data['password']
        )

        # Create the Student instance
        student = super().save(commit=False)
        student.user = user
        student.email = self.cleaned_data['email']

        # Handle webcam photo capture
        photo_data = self.cleaned_data.get('photo_data')
        if photo_data:
            format, imgstr = photo_data.split(';base64,')
            ext = format.split('/')[-1]

            # Generate filename
            filename = f"{student.admission_number}_profile.{ext}"
            filepath = os.path.join(settings.MEDIA_ROOT, 'student_photos', student.admission_number, filename)

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save the image
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(imgstr))

            # Save the photo path to the student model
            student.photo.name = f'student_photos/{student.admission_number}/{filename}'

        if commit:
            student.save()
            self.save_m2m()  # Save many-to-many relationships

        return student


class StudentUpdateForm(forms.ModelForm):
    photo_data = forms.CharField(widget=forms.HiddenInput(), required=False)

    class Meta:
        model = Student
        fields = ['full_name', 'photo', 'course', 'registered_units']
        widgets = {
            'photo': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'registered_units': forms.CheckboxSelectMultiple,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['photo'].required = False

    def save(self, commit=True):
        student = super().save(commit=False)

        # Handle webcam photo capture
        photo_data = self.cleaned_data.get('photo_data')
        if photo_data:
            format, imgstr = photo_data.split(';base64,')
            ext = format.split('/')[-1]

            # Generate filename
            filename = f"{student.admission_number}_profile.{ext}"
            filepath = os.path.join(settings.MEDIA_ROOT, 'student_photos', student.admission_number, filename)

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save the image
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(imgstr))

            # Save the photo path to the student model
            student.photo.name = f'student_photos/{student.admission_number}/{filename}'

        # Handle photo removal
        if self.cleaned_data.get('photo-clear'):
            student.photo.delete(save=False)
            student.photo = None

        if commit:
            student.save()
            self.save_m2m()

        return student
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['photo'].required = False
        # Add a custom attribute to store the current image URL
        if self.instance and self.instance.photo:
            self.fields['photo'].widget.attrs['data-current-image'] = self.instance.photo.url