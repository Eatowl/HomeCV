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

import sys
from keras.datasets import mnist
from matplotlib import pyplot


#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()


#for i in range(9):
#    pyplot.subplot(330 + 1 + i)
#    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#print(len(train_X))
#print(train_X[0])
#print("&"*80)
#print(len(train_y))
#pyplot.show()

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output > 0

def build_rand_weights(size_x, size_y):
    return 0.2 * np.random.random((size_x, size_y)) - 0.1

def build_true(true_value):
    true = np.zeros((1, 10))
    for i in range(len(true[0])):
        if i == true_value:
            true[0][i] = 1.0

    return true[0]

def neural_network(inputs, weights):    
    return np.dot(inputs, weights)


images, labels = (train_X[0:1000].reshape(1000, 28*28) / 255, train_y[0:1000])

one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = test_X.reshape(len(test_X), 28*28) / 255
test_labels = np.zeros((len(test_y), 10))
for i, l in enumerate(test_y):
    test_labels[i][l] = 1


alpha = 0.005
hidden_size = 100
pixel_per_image = 784
num_labels = 10
iterations = 300

np.random.seed(1)

weights_0_1 = build_rand_weights(pixel_per_image, hidden_size)
weights_1_2 = build_rand_weights(hidden_size, num_labels)

for iter in range(iterations):
    error, correct_cnt = (0.0, 0)
    for i in range(len(images)):

        layer_0 = images[i:i+1]
        layer_1 = relu(neural_network(layer_0, weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = neural_network(layer_1, weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = (labels[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)


    sys.stdout.write("\r"+" I:"+str(iter)+\
                     " Error:"+str(error/float(len(images)))[0:5] +\
                     " Correct:"+ str(correct_cnt/float(len(images))))

    error, correct_cnt = (0.0, 0)
    if iter % 10 == 0 or iter == iterations - 1:

        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = relu(neural_network(layer_0, weights_0_1))
            layer_2 = neural_network(layer_1, weights_1_2)

            error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
            correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

        sys.stdout.write(" Test-Err:" + str(error/float(len(test_images)))[0:5] +\
                    " Test-Acc:" + str(correct_cnt/float(len(test_images))) + "\n")

# example backpropagation

print("==========backpropagation===========")

alpha = 0.2
hidden_size = 4

def relu(x):
    return (x > 0) * x

def relu2deriv(output):
    return output > 0

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([1, 1, 0, 0])

weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(2):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)
        
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)
        layer_2_delta = walk_vs_stop[i:i+1] - layer_2
        #print(weights_1_2.T)
        #print(layer_2_delta.dot(weights_1_2.T))
        #print(relu2deriv(layer_1))
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        #print(layer_0, layer_1_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
        #print(weights_0_1)
        
    #print(layer_1_delta)
        
    '''if iteration % 10 == 9:
        print("Error: {}".format(layer_2_error))
        print(layer_2)'''