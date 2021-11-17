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

alpha = 2
pixel_per_image = 784
num_labels = 10
iterations = 200
batch_size = 128

input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16

np.random.seed(1)

def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return 1 - (output ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

def neural_network(inputs, weights):    
    return np.dot(inputs, weights)

def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)

images, labels = (train_X[0:1000].reshape(1000, 28*28) / 255, train_y[0:1000])
one_hot_labels = np.zeros((len(labels), 10))


for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = test_X.reshape(len(test_X), 28*28) / 255
test_labels = np.zeros((len(test_y), 10))
for i, l in enumerate(test_y):
    test_labels[i][l] = 1


hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for iter in range(iterations):
    correct_cnt = 0
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size), ((i+1)*batch_size))

        layer_0 = images[batch_start:batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        sects = []
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows,\
                                                  col_start, col_start + kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0]*es[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0],-1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = softmax(neural_network(layer_1, weights_1_2))

        for k in range(batch_size):
            labelset = labels[batch_start+k:batch_start+k+1]
            _inc = int(np.argmax(layer_2[k:k+1]) == np.argmax(labelset))
            correct_cnt += _inc

        layer_2_delta = (labels[batch_start:batch_end] - layer_2)\
                                 / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(l1d_reshape)
        kernels -= alpha * k_update
    
    test_correct_cnt = 0
    for i in range(len(test_images)):
        layer_0 = test_images[i:i+1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        sects = []
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0, row_start, row_start + kernel_rows,\
                                                  col_start, col_start + kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0]*es[1], -1)


        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0],-1))
        layer_2 = neural_network(layer_1, weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))


    if iter % 10 == 0:
        sys.stdout.write("\r"+" I:"+str(iter)+\
                              " Correct:"+ str(correct_cnt/float(len(images)))+\
                              " Test-Acc:" + str(test_correct_cnt/float(len(test_images))) + "\n")


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