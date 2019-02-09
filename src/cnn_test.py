from __future__ import print_function
import cv2
import numpy as np
import math
import sys
import os
import keras
import logging
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard


# input image dimensions
img_rows, img_cols = 28, 28
batch_size = 5

#Main method
def main():
      global imgMod, imgOrg
      print("Author: Marina Torres \n")
      print("This program classifies input image numbers as odd or even. \n")
      
      while(True):
          inp = raw_input('Enter a filename, press "q" to quit, or press "h" for help: ')
          if(inp == "q"):
              sys.exit(0)
          elif(inp == "h"):
              help()
          else:
            try:
              imgOrg = getImage(inp)
              imgMod = gray(imgOrg)
              imgOrg = resize(imgOrg)
              display1(imgOrg)
              imgMod = resize(imgMod)
              imgMod = grayToBin(imgMod)
              display2(imgMod)
              imgMod = np.expand_dims(imgMod,0)
              imgMod = np.expand_dims(imgMod,3)
              predict(imgMod)
            except:
              print("Please, insert a valid input")

## Image pre-processing functions

#0. Gray scale
def gray(img):
    img = img.copy()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    return imgGray

#2a. Resize the image
def resize(img):
    img = img.copy()
    imgS = cv2.resize(255 - img, (img_cols, img_rows)) 
    return imgS

#2b. Transform from grayScale to binary image
def grayToBin(img):
    (th, binary) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

#2c. Display original and binary image in separated windows
def display1(img):
    cv2.imshow('Original image',img)
    value = cv2.waitKey(50)&0xff
    if(value == ord('q')):
      cv2.destroyAllWindows()

def display2(img):
    cv2.imshow('Modified image',img)
    value = cv2.waitKey(50)&0xff
    if(value == ord('q')):
      cv2.destroyAllWindows()
    
#1. Accept as input an image of a handwritten digit
def getImage(file):
       img = cv2.imread("../data/"+file)
       return img 

#3. Classify binary image using the CNN
def predict(img):
  model = load_model('../models/model.h5')	
  print("Loaded model from disk")
  prediction = model.predict(img)

  if(prediction[0][0]>prediction[0][1]):
    print("The number is even")
  else: 
    print("The number is odd")

def help():
  print("This program requests continuously the path to an image file.")
  print("It processes the input image and outputs the class of the image")
  print("To start, insert a valid image name")
  print("Press q or Esc at any moment to quit the program")

if __name__ == "__main__": main()
