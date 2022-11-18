import cv2
import numpy as np
from FPS import *
import os




def drawRec(Color, left, top, right, bottom, label):
    cv2.rectangle(frame, (left, top), (right, bottom), Color, 1)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  Color, cv2.FILLED) 
    cv2.putText(frame, f'{label}', (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0, 0), 1)

def drawPred(classId, conf, left, top, right, bottom):
    label = '%.2f' % conf
    label = int(float(label) * 100)
    label = (str(label) + '%')
    assert (classId < len(classes))
    label = '%s:%s' % (classes[classId], label)
  

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)



classesFile = "weights";
classes = True
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "conf.cfg";
modelWeights = "w.weights";
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

width = 1920
height = 1080
ResizeFrame = 2


cap = cv2.VideoCapture('test.mp4') 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)




while cv2.waitKey(1) < 0:

    hasFrame, frame = cap.read()
    frame = cv2.resize(frame, (int(width / ResizeFrame), int(height / ResizeFrame)),
                     interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)
    Counter += 1


    cv2.imshow('Result', frame)
cap.release()
cv2.destroyAllWindows()






















