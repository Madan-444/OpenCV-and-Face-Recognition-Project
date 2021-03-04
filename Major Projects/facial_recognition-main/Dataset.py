import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join

# nameOfStudent = input("What is your??")
idOfStudent = input("What is your student ID??")
newPath = "C:/Users/Dell/Desktop/image/{}".format(idOfStudent)
# againNewpath = os.mkdir(newPath)
os.mkdir(newPath)
myPath = "C:/Users/Dell/Desktop/image/{}".format(idOfStudent)
print(myPath)


face_classifier = cv2.CascadeClassifier('C:/Users/Dell/Desktop/attendance/Major Projects/haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        

        file_name_path =  'C:/Users/Dell/Desktop/image/008/'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print('Samples Colletion Completed ')


data_path = 'C:/Users/Dell/Desktop/image/008'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Dataset Model Training Completed ")
