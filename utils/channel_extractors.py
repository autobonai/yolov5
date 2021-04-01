import cv2
import numpy as np


def extract_red(frame):
    return _extract_one_idx(frame, 2)

def extract_green(frame):
    return _extract_one_idx(frame, 1)

def extract_blue(frame):
    return _extract_one_idx(frame, 0)

def extract_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

def extract_hue(frame):
    return _extract_one_idx(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), 0)

def extract_saturation(frame):
    return _extract_one_idx(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), 1)

def extract_value(frame):
    return _extract_one_idx(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), 2)

def _extract_one_idx(frame, idx):
    return frame[..., [idx]]