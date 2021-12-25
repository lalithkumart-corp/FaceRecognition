import cv2;

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

img = cv2.imread('unknown-user-images/1640445431.jpeg')

names = ['dummy', 'lalith', 'shovanlal', 'bhavi']
font = cv2.FONT_HERSHEY_SIMPLEX

# img = cv2.resize(img, (300,300))
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

while True:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5
       )
    if(not len(faces)):
        cv2.putText(
                    img, 
                    'Face could not be detected', 
                    (0,60), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
    else:
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            # If confidence is less them 100 ==> "0" : perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(
                        img, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
            cv2.putText(
                        img, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                    ) 
        
    cv2.imshow('Image Predicted ', img)
    cv2.waitKey(5000)
    # k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    break
# cv2.destroyAllWindows() 