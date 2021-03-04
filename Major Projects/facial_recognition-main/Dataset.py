import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime

qtn = input('Are you new User ?? Yes or No').upper()
myName = input("Enter your name")
idOfStudent = input("What is your student ID??")

if qtn == 'YES':

    face_classifier = cv2.CascadeClassifier(
    'C:/Users/Dell/Desktop/attendance/Major Projects/haarcascade_frontalface_default.xml')
    
    # newPath = "C:/Users/Dell/Desktop/image/{}".format(idOfStudent)
 

    def face_extractor(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None

        for(x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face


    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = 'C:/Users/Dell/Desktop/image/045/'+str(count)+'.jpg'

            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(count), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Samples Colletion Completed ')

    
    data_path = 'C:/Users/Dell/Desktop/image/045/'
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

    print("Dataset Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('C:/Users/Dell/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))



            if confidence > 82:
                cv2.putText(image, name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==13:
            break


    cap.release()
    cv2.destroyAllWindows()


else:
    print("My name is {}".format(myName))
    data_path = 'C:/Users/Dell/Desktop/image/{}/'.format(idOfStudent)
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

    print("Dataset Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('C:/Users/Dell/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))



            if confidence > 82:
                cv2.putText(image, myName, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Face Cropper', image)

            else:
                cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==13:
            break


    cap.release()
    cv2.destroyAllWindows()

