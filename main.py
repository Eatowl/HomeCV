#!/home/denis/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.stats
import imagehash


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
    frame_sum = 0
    frame_sum_control = 0
    while(True):

        ret, frame = video.read()
        frame_count += 1

        if frame_count == 5:
            frame_count = 0
            frame_sum_control = np.mean(frame)
        #print(np.sum(frame))

        print(frame_sum_control / 100000)

        viewImage(frame, "window")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


move_detect()
