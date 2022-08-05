import os
import cv2
import csv
import numpy as np
import string 
import random
from dist import EuclideanDistTracker
from model import net

np.random.seed(42)
model_yolo_v3 = net()

class config:
    def find_center(x, y, w, h):
        x1=int(w/2)
        y1=int(h/2)
        cx = x+x1
        cy=y+y1
        return cx, cy

class Model(config):
    def __init__(self, Video):
        self.video = Video
        self.tracker = EuclideanDistTracker()
        self.temp_up_list = []
        self.temp_down_list = []
        self.confThreshold=0.7
        self.nmsThreshold= 0.7
        self.font_color = (0, 0, 255)
        self.font_size = 0.5
        self.font_thickness = 2
        self.up_list = [0, 0, 0, 0]
        self.down_list = [0, 0, 0, 0]
        self.middle_line_position = 180 
        self.up_line_position = self.middle_line_position - 15
        self.down_line_position = self.middle_line_position + 15
        self.classesFile = "coco.names"
        self.classNames = open(self.classesFile).read().strip().split('\n')
        self.required_class_index = [2, 3, 5, 7]
        self.detected_classNames = []
        self.net = model_yolo_v3.model_yolo
        self.colors = np.random.randint(0, 255, size=(len(self.classNames), 3), dtype='uint8')

    def count_vehicle(self, box_id, img):
        x, y, w, h, id, index = box_id
        # Find the center of the rectangle for detection
        center = self.find_center(x, y, w, h)
        ix, iy = center

        # Find the current position of the vehicle
        if (iy > self.up_line_position) and (iy < self.middle_line_position):

            if id not in self.temp_up_list:
                self.temp_up_list.append(id)

        elif iy < self.down_line_position and iy > self.middle_line_position:
            if id not in self.temp_down_list:
                self.temp_down_list.append(id)
                
        elif iy < self.up_line_position:
            if id in self.temp_down_list:
                self.temp_down_list.remove(id)
                self.up_list[index] = self.up_list[index]+1

        elif iy > self.down_line_position:
            if id in self.temp_up_list:
                self.temp_up_list.remove(id)
                self.down_list[index] = self.down_list[index] + 1

        # Draw circle in the middle of the rectangle
        cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here


    def count_vehicle(self, box_id, img):
        x, y, w, h, id, index = box_id

        # Find the center of the rectangle for detection
        center = config.find_center(x, y, w, h)
        ix, iy = center
        
        # Find the current position of the vehicle
        if (iy > self.up_line_position) and (iy < self.middle_line_position):

            if id not in self.temp_up_list:
                self.temp_up_list.append(id)

        elif iy < self.down_line_position and iy > self.middle_line_position:
            if id not in self.temp_down_list:
                self.temp_down_list.append(id)
                
        elif iy < self.up_line_position:
            if id in self.temp_down_list:
                self.temp_down_list.remove(id)
                self.up_list[index] = self.up_list[index]+1

        elif iy > self.down_line_position:
            if id in self.temp_up_list:
                self.temp_up_list.remove(id)
                self.down_list[index] = self.down_list[index] + 1

        # Draw circle in the middle of the rectangle
        cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    
    def postProcess(self, outputs, img):
        height, width = img.shape[:2]
        boxes = []
        classIds = []
        confidence_scores = []
        detection = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if classId in self.required_class_index:
                    if confidence > self.confThreshold:
                        # print(classId)
                        w, h = int(det[2] * width), int(det[3] * height)
                        x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                        boxes.append([x, y, w, h])
                        classIds.append(classId)
                        confidence_scores.append(float(confidence))

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, self.confThreshold, self.nmsThreshold)
        # print(classIds)
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in self.colors[classIds[i]]]
            name = self.classNames[classIds[i]]
            self.detected_classNames.append(name)
            # Draw classname and confidence score 
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, self.required_class_index.index(classIds[i])])

        # Update the tracker for each object
        boxes_ids = self.tracker.update(detection)
        for box_id in boxes_ids:
            self.count_vehicle(box_id, img)
    
    def realTime(self):
        cap = cv2.VideoCapture(self.video)
        output_filename = 'output_'+''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))+'.mp4'
        out = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'MP4V'),20,(640, 360))
        while True:
            success, img = cap.read()
            if img is None:
                break
            ih, iw, channels = img.shape
            img  = cv2.resize(img,(640, 360))
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)

            # Set the input of the network
            self.net.setInput(blob)
            layersNames = self.net.getLayerNames()
            outputNames = [(layersNames[i[0] - 1]) for i in self.net.getUnconnectedOutLayers()]
            
            # Feed data to the network
            outputs = self.net.forward(outputNames)

            # Find the objects from the network output
            self.postProcess(outputs, img)

            # Draw the crossing lines
            cv2.line(img, (0, self.up_line_position), (int(iw), self.up_line_position), (255, 0, 255), 1)
            cv2.line(img, (0, self.middle_line_position), (int(iw), self.middle_line_position), (0, 0, 255), 1)
            cv2.line(img, (0, self.down_line_position), (int(iw), self.down_line_position), (0, 0, 255), 1)

            # Draw counting texts in the frame
            cv2.putText(img, "Car:        " + str(self.up_list[0]) + "     " + str(self.down_list[0]), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            cv2.putText(img, "Bike:       " + str(self.up_list[1]) + "    " + str(self.down_list[1]), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            cv2.putText(img, "Bus:        " + str(self.up_list[2]) + "     " + str(self.down_list[2]), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)
            cv2.putText(img, "Truck:      " + str(self.up_list[3]) + "     " + str(self.down_list[3]), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_color, self.font_thickness)

            out.write(img) 
            if cv2.waitKey(1) == ord('q'):
                break

        # Write the vehicle counting information in a file and save it
        out.release()
        cap.release()
        data_filename = 'data_'+''.join(random.choices(string.ascii_uppercase + string.digits, k = 7))+'.csv' 
        # f1 = open('\\static\\results\\'+data_filename, 'w+')
        # cwriter = csv.writer(f1)
        # cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        # self.up_list.insert(0, "Up")
        # self.down_list.insert(0, "Down")
        # cwriter.writerow(self.up_list)
        # cwriter.writerow(self.down_list)
        # f1.close()
        return data_filename, output_filename