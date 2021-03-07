import csv
from datetime import datetime

myName = input("What is your name ??")
idOfStudent = input("What is your student id")


with open('attendance.csv', mode='r+') as attendance_file:
    attendance_writer = csv.writer(attendance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    myDataList = attendance_file.readlines()
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    attendance_writer.writerow([ myName, dtString, idOfStudent])
