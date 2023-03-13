import cv2
import numpy as np
from twilio.rest import Client
import matplotlib.pyplot as plt
net = cv2.dnn.readNetFromDarknet("yolov3_testing.cfg",r"rajeshfinal2023.weights")
### Change here for custom classes for trained model
import threading

# Define the camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480

classes = ['Rajesh']

cap = cv2.VideoCapture(0)
def get_google_maps_link(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

lat = 37.7749
lon = -122.4194

def dis():
    account_sid = "AC3f01bc3084724dbd4a97bd3236b398a4"
    auth_token = "a3c4c55183bddb7425cc43ced22a6e2c"
    client = Client(account_sid, auth_token)
    print("location has been sent to your phone")
    google_maps_link = get_google_maps_link(lat, lon)
    message = client.messages.create(
        body=" I am at: " + google_maps_link,
        from_="+18302678502",
        to="+917397571872"
    )
    print(message.sid)

# Set the camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Define a function to capture and process frames
def process_frames():
    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (640, 480))
            hight, width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

            net.setInput(blob)
            output_layers_name = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_name)

            boxes = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > 0.8:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * hight)
                        w = int(detection[2] * width)
                        h = int(detection[3] * hight)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    dis()
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)

            cv2.imshow('img', img)


        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


def display_frames():
    while True:
        ret, frame = cap.read()
        if ret: cv2.imshow('Raw Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

threading.Thread(target=process_frames).start()
# threading.Thread(target=display_frames).start()
