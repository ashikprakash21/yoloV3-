import cv2
import numpy as np

import detect_obj_2
import blob_img

image = detect_obj_2.image_BGR
h,w = image.shape[:2]

blob = blob_img.blob

with open('prog_2/YOLO-3-OpenCV/yolo-coco-data/coco.names') as f:
    labels = [label.strip() for label in f]

# print("A"*100)
# print(labels)
# print(len(labels))




network = cv2.dnn.readNetFromDarknet('prog_2/YOLO-3-OpenCV/yolo-coco-data/yolov3.cfg',
                                     'prog_2/YOLO-3-OpenCV/yolo-coco-data/yolov3.weights')



layer_names_all = network.getLayerNames()

# print("B"*100)
# print(layer_names_all)
# for j in layer_names_all:
#     print(j)
    

output_layers = [layer_names_all[i-1] for i in network.getUnconnectedOutLayers()]
print(output_layers)


probability_min = 0.9

threshold = 0.3

colors = np.random.randint(0, 255, size=(len(labels),3), dtype='uint8')


# print("C"*100)
# print(output_layers)


# print("D"*100)
# print(colors.shape)
# print(colors[0])


import time

print("F"*100)
#forward pass filtering
network.setInput(blob)
start = time.time()
output_from_network = network.forward(output_layers)
end = time.time()

# print("time taken:{}".format(start-end))



bounding_boxes = []
confidences =[]
class_numbers = []




#result from output_from_network
# print("I"*100)
for result in output_from_network:
    # print(result)
    for detected_objs in result:
        # print(detected_objs)
        
        #getting 80 classes probablity for current detected object.
        scores = detected_objs[5:]
        
        #getting index of class with highest probabaility
        current_class = np.argmax(scores)
        
        #getting probability of current detected object.
        confidence_current = scores[current_class]
        # print(confidence_current)


        if confidence_current>probability_min:
            
            box_current = detected_objs[0:4] * np.array([w,h,w,h])
            
            #from YOLO data format, we obtain top lefty coordinates
            x_centre, y_centre, box_width, box_height = box_current
            
            x_min = int(x_centre - (box_width/2))
            y_min = int(y_centre - (box_height/2))
            
            
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(current_class)
            
            
            
            
            
#implementing non-maximum suppression( NMS )
result = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_min, threshold)


# print("L"*100)
# print(confidences)
# print(class_numbers)
# print(result.flatten())         




#drawing bounding boxes


'''
### we giving each item we got from NMS result into the class_numbers we got after detecting high
### confident bounding boxes
##we will get the labels of detected objects
'''


#setting a counter
counter = 1

if len(result) > 0:
    
    for i in result.flatten():
        # print("M"*100)
        # print(i)
        print('object {} : {}'.format(counter, labels[int(class_numbers[i])]))
        # print(class_numbers[i])
        # print(labels[int(class_numbers[i])])
        
        counter += 1
        
        
        #getting the cordinates of current bounding boxes
        # print("O"*100)
        # print(bounding_boxes[i])
        x_min, y_min = bounding_boxes[1][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
        
        # print("S"*100)
        #preparing colors for current bounding boxes
        #and converting it(array) to list
        curent_bb_color = colors[class_numbers[i]].tolist()
        
        
        #drawing bounding boxes on org image
        cv2.rectangle(image, (x_min, y_min), (x_min+box_width, y_min+box_height), curent_bb_color, 2)
        
        #preparing text with label and confidence for current bounding boxes
        text_box_current = '{}:{:.4f}'.format(labels[int(class_numbers[i])],confidences[i])
        
        #putting text with label and confidence on the org image
        cv2.putText(image, text_box_current, (x_min,y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    curent_bb_color, 2)



print()
print("total objects detected :", len(bounding_boxes))
print("no. of objects left after non-maximim suppression : ", counter - 1)
        
        
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyWindow("Detections")

