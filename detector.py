# import necessary dependencies
import os
import cv2
import pickle
import imutils
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class Detector:
    def __init__(self):
        # load serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.join(os.getcwd(), "face_detector/deploy.prototxt.txt")
        modelPath = os.path.join(os.getcwd(), "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load the liveness detector model and label encoder from disk
        print("[INFO] loading liveness detector...")
        self.model = load_model(os.path.join(os.getcwd(), "model/liveness_mobile.model"))
        self.le = pickle.loads(open(os.path.join(os.getcwd(), "le.pickle"), "rb").read())

    def predict(self, true_img, test_img, confidence_threshold):
        true_img = imutils.resize(true_img, width=600)
        test_img = imutils.resize(test_img, width=600)

        (h, w) = test_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(test_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        liveness_result = False

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > confidence_threshold:
                # compute the (x, y)-coordinates of the bounding box for
			    # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # extract the face ROI and then preprocess it in the exact same manner as our training data
                face = test_img[startY:endY, startX:endX]
                face = cv2.resize(face, (224, 224))
                face = face.astype("float") /255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector model to determine if the face is 'real' or 'fake'
                preds = self.model.predict(face)[0]
                j = np.argmax(preds)
                label = self.le.classes_[j]

                real_face= bool(label)


                
                # if this is a real live face, we change the liveness_result to True and break out of the loop
                if real_face:
                    liveness_result = True
                    break
        
        # if there is a live face in the test image, we verify it against our true image
        if liveness_result:
            return DeepFace.verify(true_img, test_img)['verified']

        # if nothing matches, we return False
        return False


# just for testing purposes
if __name__ == "__main__":
    detector = Detector()
    a = cv2.imread("test_img/1.jpeg")
    b = cv2.imread("test_img/2.jpeg")
    result = detector.predict(a, b, 0.8)
    print(result)
