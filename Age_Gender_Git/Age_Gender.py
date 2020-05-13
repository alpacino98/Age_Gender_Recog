import cv2
import numpy as np

MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']


def initialize_caffe_model():
    print('Loading models...')
    ageProto = "/users/AlphaPro/Desktop/Windows/Age_Gender_Recog/age_deploy.prototxt.txt"
    ageModel = "/users/AlphaPro/Desktop/Windows/Age_Gender_Recog/age_net .caffemodel"
    age_net = cv2.dnn.readNet(ageModel, ageProto)
    genderProto = "/users/AlphaPro/Desktop/Windows/Age_Gender_Recog/gender_deploy.prototxt.txt"
    genderModel = "/users/AlphaPro/Desktop/Windows/Age_Gender_Recog/gender_net.caffemodel"
    gender_net = cv2.dnn.readNet(genderModel, genderProto)



    return (age_net, gender_net)


def facechop(age_net, gender_net, image):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    #img = cv2.imread(image)

    minisize = (image.shape[1], image.shape[0])
    miniframe = cv2.resize(image, minisize)

    faces = cascade.detectMultiScale(miniframe)
    print(faces)
    if(faces == ()):
        print("No faces")
        age = 0
        gender = 0
        return image, age, gender
    else:
        for f in faces:
            x, y, w, h = [v for v in f]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face_img = image[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN, swapRB=False)

            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            overlay_text = (gender + ' , ' + age)
            cv2.putText(image, overlay_text, (x-10, y), 1, 1, (0, 0, 0), 1, cv2.LINE_AA)

            '''global sub_face
            sub_face = image[y:y + h, x:x + w]

            f_name = image.split('/')
            f_name = f_name[-1]'''

                # print ("Writing: " + image)
        # cv2.imshow(image, img)
    return image, age, gender



age_net, gender_net = initialize_caffe_model()


video = cv2.VideoCapture(0)
#video = cv2.VideoCapture(0)
a = 0
boo = True
c = 0
while boo:
    a = a + 1
    check, frame = video.read()
    print(np.asarray(frame).shape)
    frame = cv2.resize(frame, (227 ,227 ))
    print(np.asarray(frame).shape)
    gray, Age, gender = facechop(age_net, gender_net, frame)
    gray = cv2.resize(gray, (1000, 1000))


    print(Age)
    print(gender)
    #cv2.putText(frame, gray, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("GENDER", gray)

    # cv2.waitKey(0)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break
print(a)

video.release()
cv2.destroyAllWindows
