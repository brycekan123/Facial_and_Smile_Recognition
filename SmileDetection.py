import face_recognition
import cv2

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("smilingandfrowning.webp")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml') 

#detects faces
faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5) 

#sorting faces from left to right
faces = sorted(faces, key=lambda face: face[0])

smilecounter = 0
facecounter = 0
for (x, y, width, height) in faces: 
    facecounter+=1
    
    #Draws rectange around face based on parameters.
    x_offset = 12
    y_offset = 25
    cv2.rectangle(image, (x+x_offset, y-y_offset), ((x + width-x_offset), 
                                                    (y + height)),(255, 255, 255), 2) 
    cv2.putText(image, f'Face{facecounter}', (x + 40, y-y_offset), 
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    

    #Crop out each individual face
    IndividualFace = image[y:y + height, x:x + width] 
    
    #detects smiles
    smiles = smile_cascade.detectMultiScale(IndividualFace, scaleFactor=1.9, minNeighbors=10) 
    #eyes = eye_cascade.detectMultiScale()

    for (smile_x, smile_y, smile_width, smile_height) in smiles: 
        smilecounter +=1
        #Draws rectange around smile based on parameters
        cv2.rectangle(IndividualFace, (smile_x, smile_y), ((smile_x + smile_width), 
                                                           (smile_y + smile_height)), (0, 0, 255), 2)         
        text_y = max(smile_y - 5, 0)  # Ensure text_y doesn't go out of the image
        cv2.putText(IndividualFace, f'Smile{smilecounter}', (smile_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

print(f"Number of Faces Detected: {facecounter}")
print(f"Number of Smiles Detected: {smilecounter}")
cv2.imshow('image',image)  
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 
