from os import name
from teachable_machine_lite import TeachableMachineLite
import cv2 as cv
import pickle
import face_recognition
from time import sleep
import time
import RPi.GPIO as GPIO

cap = cv.VideoCapture(0)

model_path = 'model.tflite'
image_file_name = "frame.jpg"
labels_path = "labels.txt"

tm_model = TeachableMachineLite(model_path=model_path, labels_file_path=labels_path)

# Loading the characteristics of trained faces from encodings.pickle file model
encodingsP = "encodings.pickle"

cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings
print("loading encodings + face detector to OpenCV classifier...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv.CascadeClassifier(cascade)

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

buzzer_pin = 32

GPIO.setup(buzzer_pin, GPIO.OUT)

def recognize_faces(frame):
    # convert frame from BGR to grayscale for face detection
    # Convert frame from BGR to RGB for face recognition
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the frame to the trained faces
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"  # if face is not recognized, then print Unknown

        # check to see if we have predefined person
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

            # If someone in your dataset is identified, print their name on the screen
            print(name)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv.rectangle(frame, (left, top), (right, bottom),
                     (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv.putText(frame, name, (left, y), cv.FONT_HERSHEY_SIMPLEX,
                   .8, (255, 0, 0), 2)
    return names

def buzz(pitch, duration):
    period = 1.0 / pitch
    delay = period / 2
    cycles = int(duration * pitch)
    for i in range(cycles):
        GPIO.output(buzzer_pin, True)
        time.sleep(delay)
        GPIO.output(buzzer_pin, False)
        time.sleep(delay)
i = 0
while True:
    ret, frame = cap.read()
    cv.imshow('Cam', frame)
    
    if i > 4:
        i=0
        cv.imwrite(image_file_name, frame)
        results = tm_model.classify_frame(image_file_name)
        label = results["label"]
        print(label)
        if label == "Yasmeenmask" or label == "Aseelmask":
            print("allowed")
        if label == "Yasmeen" or label == "Aseel":
            print("violating code")
            people = recognize_faces(frame)
            
            for name in people:
                print(f"Warning!!! {name} violated safety code")
                
            buzz(440, 0.5) # buzz at 440 Hz for 0.5 seconds
      
    else:
        i+=1
        sleep(0.09)
    
    k = cv.waitKey(1)
    if k% 255 == 27:
        # press ESC to close camera view.
        break

    

