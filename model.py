import cv2

class net:
    def __init__(self):
        self.modelConfiguration = 'yolov3.cfg'
        self.modelWeigheights = 'yolov3.weights'
        self.model_yolo = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeigheights)
        self.model_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)