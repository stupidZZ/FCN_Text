#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sys import argv
from matplotlib import pyplot as plt
import cv2
import cv
import numpy as np
from compiler.ast import flatten

def computeMSER(grayImg):
    mser = cv2.MSER(1, 20, 200000, 1, 0.01, 200, 1.01, 0.003, 5)
    regions = mser.detect(grayImg)
    regionNum = len(regions)
    boxes = np.zeros((regionNum, 4))
    minAreaBoxes = np.zeros((regionNum, 8))

    for i in range(0, regionNum):
        region = np.array(regions[i])
        minXY = np.min(region, 0)
        maxXY = np.max(region, 0)
        boxes[i, :] = [minXY[0] + 1, minXY[1] + 1, maxXY[0] - minXY[0] + 1, maxXY[1] - minXY[1]+1]
        minAreaBox = np.int0(cv2.cv.BoxPoints(cv2.minAreaRect(regions[i])))
        minAreaBox = np.reshape(minAreaBox, (8))
        minAreaBoxes[i, :] = minAreaBox   

    return boxes, minAreaBoxes

if __name__ == '__main__':
    imgPath = argv[1]
    resPath = argv[2]

    img = cv2.imread(imgPath)

    # Compute MSER for the image
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mserBoxes, minAreaBoxes = computeMSER(grayImg)
    
    # mserBoxes = np.append(mserBoxes1, mserBoxes2, 0)


    # Construct the output strings
    mserNum = mserBoxes.shape[0]
    outputStrings = []
    for i in range(0, mserNum):
        outputString = ""
        #outputString = '%d %d %d %d\r\n' % (mserBoxes[i, 0], mserBoxes[i, 1], mserBoxes[i, 2], mserBoxes[i, 3])
        for j in range(0, 8):
            outputString = outputString + '%d ' % (minAreaBoxes[i, j])

        outputString = outputString + "\r\n"
        outputStrings.append(outputString)

    fo = open(resPath, 'w')
    fo.writelines(outputStrings)

    fo.close()
    

