import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import math


def ellipse_fitting(image):
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    axis = []

    for c in contours:
        e = cv2.fitEllipse(c)
        print(e)
        xc = e.center.x
        yc = e.center.y
        a = e.size.width / 2
        b = e.size.height / 2
        axis.append([a, b])

    return axis


def rectangle_fitting(image):
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    length = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        length.append(w)

    return length


def head_circumference(a, b):
    h = ((a - b) ** 2) / ((a + b) ** 2)
    hc = math.pi * ((a + b) * (1 + ((3 * h) / (10 + math.sqrt(4 - 3 * h)))))

    return hc


def abdominal_circumference(a, b):
    ac = math.pi * ((3 * (a + b)) - math.sqrt((3 * a + b) * (a + 3 * b)))

    return ac


def gestational_age(femur_length):
    ga = 1.863 + (6.280 * femur_length) - (0.211 * (femur_length ** 2))

    return ga
