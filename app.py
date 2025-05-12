import cv2
import cv2.data
import matplotlib.pyplot as plt

imagePath = 'test.jpg'

# Reading the img using cv2

img = cv2.imread(imagePath)

print(img.shape) # if the color is colored the result will end with 3 at the end (1 , 1 , 3)

# to improve computing we convert the image to grayscale 
gray_image = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

print(gray_image.shape) # no 3 at the end of the dimension cuz it's not colored 


# Loading the pre-trained classifier Haar Caascade there is also a lot of pre-trained models on github https://github.com/opencv/opencv/tree/master/data/haarcascades

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # thi xml file is designed specifically for detecting front face in visual input 
) 

# performing the face detection 

face = face_classifier.detectMultiScale(
    gray_image , scaleFactor=1.1 , minNeighbors= 5, minSize=(40,40)
)

# detectMultiScale - used to identify faces of different sizes in the input image.
# scaleFactor - used to scale down the size of the input image to make it easier for the algorithm to detect larger faces. In this case, we have specified a scale factor of 1.1, indicating that we want to reduce the image size by 10%.
# minNeighbors - used to slide windows (window let's say a small rectangle scanner) on the picture to detect faces in it 


# Drawing a bounding box 

for (x,y,w,h) in face : 
    cv2.rectangle(img , (x,y) , (x + w, y + h) , (0,255,0) , 4)


# displaying the image 

# converting it from grayscale to RGB 

img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

# importing matplotlib to display the image

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')

plt.show()