# import cv2
# import numpy as np
# import face_recognition
# import os
#
# path = 'ImageAttendace'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)
# for cls in myList:
#     curImg = cv2.imread(f'{path}/{cls}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cls)[0])
# print(classNames)
#
# def findencodings(images):
#     encodeList= []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
# encodeListKnown = findencodings(images)
# print('Encoding Complete')
#
# cap = cv2.VideoCapture(0)
# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
#
#     for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         print(faceDis)
#
# # print(len(encodeListKnown))
#
# # faceLocTest = face_recognition.face_locations(imgTest)[0]
# # encodeTest = face_recognition.face_encodings(imgTest)[0]
# # cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
# #
# # # Compare the faces
# # results = face_recognition.compare_faces([encodeElon], encodeTest)
# # faceDis = face_recognition.face_distance([encodeElon], encodeTest)
# # print(results, faceDis)  # Should return [False] for different faces
# # cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the folder containing images
path = 'ImageAttendace'
images = []
classNames = []
myList = os.listdir(path)
print("Image List:", myList)

# Load images and extract class names
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print("Class Names:", classNames)

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print(f"Face not detected in one of the images. Skipping...")
    return encodeList

# Function to mark attendance
def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.strip().split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")

# Find encodings for known faces
print("Encoding faces. Please wait...")
encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        # Resize and convert frame for faster processing
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces and encodings in the current frame
        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        if not encodeCurFrame:
            print("No faces detected in the current frame.")
            continue

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            # Compare the current face with known encodings
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            # Identify the best match
            matchIndex = np.argmin(faceDis) if matches else -1
            if matchIndex != -1 and matches[matchIndex]:
                name = classNames[matchIndex].upper()

                # Draw bounding box and label
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]  # Scale back to original size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
            else:
                print("No Match Found")

        # Display the webcam feed
        cv2.imshow('Webcam', img)

        # Break the loop with 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
