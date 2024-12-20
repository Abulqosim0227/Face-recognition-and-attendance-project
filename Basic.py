import cv2
import numpy as np
import face_recognition

# Load and convert the images
imgElon = face_recognition.load_image_file("ImageBasic/Elon.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("ImageBasic/Bill Gates.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Locate and encode the first face (Elon)
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# Locate and encode the second face (Test Image)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Compare the faces
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)  # Should return [False] for different faces
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
# Display the images
cv2.imshow("Elon", imgElon)
cv2.imshow("Test Image", imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
