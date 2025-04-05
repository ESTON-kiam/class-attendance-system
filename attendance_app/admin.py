from django.contrib import admin
from .models import Course, Unit, Student, Attendance


class UnitInline(admin.TabularInline):
    model = Unit
    extra = 1


class CourseAdmin(admin.ModelAdmin):
    inlines = [UnitInline]
    list_display = ('name', 'code', 'created_at')
    search_fields = ('name', 'code')


class UnitAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'course')
    list_filter = ('course',)
    search_fields = ('name', 'code')


class StudentAdmin(admin.ModelAdmin):
    list_display = ('admission_number', 'full_name', 'course', 'created_at')
    list_filter = ('course',)
    search_fields = ('admission_number', 'full_name')
    filter_horizontal = ('registered_units',)


class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('student', 'unit', 'date', 'time', 'is_present')
    list_filter = ('unit', 'date', 'is_present')
    search_fields = ('student__admission_number', 'student__full_name')


admin.site.register(Course, CourseAdmin)
admin.site.register(Unit, UnitAdmin)
admin.site.register(Student, StudentAdmin)
admin.site.register(Attendance, AttendanceAdmin)
