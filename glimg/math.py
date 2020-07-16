import math as m
import numpy as np

#------------------- triangle -------------------#
def to_angle(rad):
    return rad/np.pi*180.

def to_rad(angle):
    return angle/180.*np.pi

def cos(angle):
    rad = to_rad(angle)
    return m.cos(rad)

def sin(angle):
    rad = to_rad(angle)
    return m.sin(rad)

def tan(angle):
    rad = to_rad(angle)
    return m.tan(rad)

def acos(val):
    rad = m.acos(val)
    return to_angle(rad)

def asin(val):
    rad = m.asin(val)
    return to_angle(rad)

def atan(val):
    rad = m.atan(val)
    return to_angle(rad)