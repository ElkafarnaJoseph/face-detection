import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# access the webcam 
video_capture = cv2.VideoCapture(0) # 0 is for default , if u have multiple cameras u can change it accordingly 

# identifying Faces in video stream 

def bounding_box_detect(vid):
    gray_image = cv2.cvtColor(vid , cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

# creating a loop for Real time detection 

while True :
    result , video_frame = video_capture.read() # reading the frames of the vid 
    if result is False : 
        break # terminate the loop if the frames are not readable 

    faces = bounding_box_detect(
        video_frame
    ) # applying the function into the vid frames 

    cv2.imshow(
        'Face Detection Project' , video_frame
    ) # display the proceed frame in a window named Face Detection Project

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()