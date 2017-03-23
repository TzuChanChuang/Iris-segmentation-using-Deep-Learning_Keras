from __future__ import print_function

import os
import numpy as np

import cv2

if __name__ == '__main__':
	test_result = np.load('imgs_test_result.npy');
	np.savetxt("test_result.csv", test_result, delimiter=",")

