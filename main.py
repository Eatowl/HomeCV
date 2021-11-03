#!/home/denis/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.stats


def viewImage(image, name_of_window):

    cv2.imshow(name_of_window, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def cropping(img):

    crop_img = img[0:720, 0:643]

    return crop_img

def mean_confidence_interval(data, confidence=0.95):
    #a = 1.0 * np.array(data)
    a = data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def color_image():
    #image = cv2.imread("whiteSheet.jpg")
    #image = cv2.imread("redStar4.jpg")
    image = cv2.imread("redRunes.jpg")
    #rgb_img = cv2.cvtColor(imgorig, cv2.COLOR_BGR2RGB)

    image = cropping(image)

    print("Разрешение изображения: {} на {}".format(len(image),len(image[0])))

    print(image[0][0])
    print(image[1][1][0], image[1][1][1], image[1][1][2])

    print(type(image[0][0]))
    print(type(image))

    find_red_image = np.copy(image)

    print(type(find_red_image))

    count_red = 0

    minX = [960, 0]
    minY = [0, 720]
    maxX = [0, 0]
    maxY = [0, 0]

    red_list = []

    for y in range(len(image)):
        for x in range(len(image[0])):
            procent = (int(image[y][x][0]) + int(image[y][x][1]) + int(image[y][x][2])) / 100 

            #print(procent)

            blue_procent = int(image[y][x][0]) / procent
            green_procent = int(image[y][x][1]) / procent
            red_procent = int(image[y][x][2]) / procent

            if (blue_procent + green_procent) - red_procent < 0:
                count_red += 1
                print("Процент R-G-B в пикселе X = {} Y = {}:\nR = {}\nG = {}\nB = {}".format(x,
                                                                                              y,
                                                                                              red_procent,
                                                                                              green_procent,
                                                                                              blue_procent))

                find_red_image[y][x] = [255, 0, 0]
                red_list.append([y, x])

                if minX[0] > x: minX = [x, y]
                if minY[1] > y: minY = [x, y]
                if maxX[0] < x: maxX = [x, y]
                if maxY[1] < y: maxY = [x, y]

    print(count_red)
    print("minX ", minX)
    print("minY ", minY)
    print("maxX ", maxX)
    print("maxY ", maxY)

    valY = int(maxY[1]) - int(minY[1])
    valX = int(maxX[0]) - int(minX[0])

    print(valX, valY)


    for i in range(minX[0], valX):
        find_red_image[minX[0]][i - 10] = [0, 70, 0]

    #for i in range(maxX[0], valX): 
    #   find_red_image[maxX[0]][i + 10] = [0, 70, 0]

    for i in range(minY[1], valY):          
        find_red_image[i - 10][minY[1]] = [0, 70, 0]

    #for i in range(maxX[0], valX):        
    #   find_red_image[i + 10][maxY[1]] = [0, 70, 0]
            #print("Процент R-G-B в пикселе:\nR = {}\nG = {}\nB = {}".format(red_procent,
            #                                                               green_procent,
            #                                                               blue_procent))


    print(red_list)
    #m, mh, mH = mean_confidence_interval(red_list)
    #print(m)
    #print(mh)
    #print(mH)
    print("=====")
    print(count_red)

    viewImage(find_red_image, "Window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#color_image()


def move_detect():
    video = cv2.VideoCapture(0)

    frame_count = 0
    frame_prev = 0
    frame_sum_control = 0
    detect_count = 0
    while(True):

        ret, frame = video.read()
        frame_count += 1

        if frame_count == 5:
            frame_count = 0
            frame_prev = frame_sum_control
            frame_sum_control = np.mean(frame)
        #print(np.sum(frame))

        if frame_sum_control - frame_prev > 0.5:
            detect_count += 1
            print(frame_sum_control, frame_prev)
            print("Detect move: {}".format(detect_count))

        viewImage(frame, "window")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


#move_detect()


# MNIST test neural networks

#import numpy.random
from keras.datasets import mnist
from matplotlib import pyplot


#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors
#print(type(train_X))
#print('Y_train: ' + str(train_y.shape))
#print('X_test:  '  + str(test_X.shape))
#print('Y_test:  '  + str(test_y.shape))

#for i in range(9):
#    pyplot.subplot(330 + 1 + i)
#    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#print(len(train_X))
#print(train_X[0])
#print("&"*80)
#print(len(train_y))
#pyplot.show()


def ele_mul(number, vector):
    output = np.zeros((1, len(vector[0])))

    for i in range(len(vector[0])):
        output[0][i] = number * vector[0][i]
        
    return output[0]

def build_rand_weights():
    return np.random.default_rng(42).random((784, 10))

def build_true(true_value):
    true = np.zeros((1, 10))
    for i in range(len(true[0])):
        if i == true_value:
            true[0][i] = 1

    return true[0]

def neural_network(inputs, weights):    
    return weights.dot(inputs)

def outer_prod(inputs, delta):
    output = np.zeros((784, 10))
   
    for i in range(len(inputs)):
        output[i] = ele_mul(inputs[i], delta)
 
    return output

alpha = 0.01
weights = build_rand_weights()
error = np.zeros((1, 10))
delta = np.zeros((1, 10))


# Take the first image and translate it into a vector
input = train_X[0].ravel()
true = build_true(train_y[0])

for iter in range(9):

    weightsT = weights.transpose()
    pred = neural_network(input, weightsT)

    for i in range(len(pred)):

        error[0][i] = (pred[i] - true[i]) ** 2
        delta[0][i] = pred[i] - true[i]


    weights_delta = outer_prod(input, delta)
    weights = weights - (weights_delta * alpha)

    print("iter {}". format(iter + 1))
    print("pred: {}".format(pred))
    print("error: {}".format(error))
    print("delta: {}".format(delta))
    print('weights: {}'.format(weights))
    print('weights_delta: {}'.format(weights_delta))
    print("*"*80)

