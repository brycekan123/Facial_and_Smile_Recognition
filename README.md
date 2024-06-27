# Face and Smile Detection on Single Photo
In this program, I create a face and smile detection script using OpenCV and face_recognition libraries

```
import face_recognition
import cv2
```
# Obtaining Haarcascade Data for Face and Smile detection
```
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') 
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml') 
```

# Drawing Rectangle and Applying Text around Face
  An Offset was required to align the rectangles without overlapping each other
  
  Using face haarcascades data for face detection

```  
for (x, y, width, height) in faces: 
    facecounter+=1
    
    #Draws rectange around face based on parameters.
    x_offset = 12
    y_offset = 25
    cv2.rectangle(image, (x+x_offset, y-y_offset), ((x + width-x_offset), 
                                                    (y + height)),(255, 255, 255), 2) 
    cv2.putText(image, f'Face{facecounter}', (x + 40, y-y_offset), 
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
```
<img width="705" alt="Screenshot 2024-06-27 at 12 47 33 AM" src="https://github.com/brycekan123/Facial_and_Smile_Recognition/assets/119905092/ba7ebd1d-eff1-4ec7-9455-5ca27c2677aa">



 # Drawing Rectangle and Applying Text around Smile   
 Using smile haarcascades data for smile detection
 
 Crop out each individual face for smile detection within individual face
 ```
    IndividualFace = image[y:y + height, x:x + width]
    smiles = smile_cascade.detectMultiScale(IndividualFace, scaleFactor=1.9, minNeighbors=10) 
    for (smile_x, smile_y, smile_width, smile_height) in smiles: 
        smilecounter +=1
        cv2.rectangle(IndividualFace, (smile_x, smile_y), ((smile_x + smile_width), 
                                                           (smile_y + smile_height)), (0, 0, 255), 2)         
        text_y = max(smile_y - 5, 0)  # Ensure text_y doesn't go out of the image
        cv2.putText(IndividualFace, f'Smile{smilecounter}', (smile_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
```

<img width="704" alt="Screenshot 2024-06-27 at 12 47 01 AM" src="https://github.com/brycekan123/Facial_and_Smile_Recognition/assets/119905092/2bb0dd96-7063-4c30-9f0a-6a5e49b3ff73">

# Counting Number of Faces and Smiles Detected in given photo

```
print(f"Number of Faces Detected: {facecounter}")
print(f"Number of Smiles Detected: {smilecounter}")
```
<img width="258" alt="Screenshot 2024-06-27 at 12 46 43 AM" src="https://github.com/brycekan123/Facial_and_Smile_Recognition/assets/119905092/9c50aafd-d9cc-4751-8c95-d33a02d7becf">
