import sys
from math import floor
from Utilities.ex2_utils import *

import cv2

'''
Function:
    x^(x+1) = (sum(1->N) x_i * w_i * g(||(x^(k)-x_i) / (h)||^2) )
             -----------------------------------------------------
              (sum(1->N) w_i * g(||(x^(k)-x_i) / (h)||^2)
Variables:
    x^(k) = Current position x,y
    x^(k+1) = New position x,y
    x_i = Coordinates within the window:
        for X direction:            for Y direction:
        [[ -2, -1, 0, 1, 2 ]        [[ -2, -2, -2, -2, -2 ]
         [ -2, -1, 0, 1, 2 ]         [ -1, -1, -1, -1, -1 ]
         [ -2, -1, 0, 1, 2 ]         [ 0,   0,  0,  0,  0 ]
         [ -2, -1, 0, 1, 2 ]         [ 1,   1,  1,  1,  1 ]
         [ -2, -1, 0, 1, 2 ]]        [ 2,   2,  2,  2,  2 ]]

    w_i = Function value in x_i
    g(x) = Kernel derivative -k'(x)
        NOTE: Given a convolution kernel H, what is the corresponding MS kernel K:
            -> Perform change of variables r = ||x_i-x||^2
            -> Rewrite H(x_i-x) => h(||x_i-x||^2) => h(r)
            -> Kernel must satisfy h'(r) = -c k(r)
    h = Size of kernel (h x h)
'''


def find_max(arr):
    max_x = np.where(arr == np.amax(arr))[0][0]
    max_y = np.where(arr == np.amax(arr))[1][0]
    max_val = np.amax(arr)
    return (max_x, max_y), max_val


def MS_modeSeeking(img=generate_responses_1(), k_sz=(5, 5), center=(50, 50), thresh=0.02, n_iters=10000, display=False):
    img[img == 0] = sys.float_info.epsilon
    if display:
        print(f"Maximum value and its location with 'np.amax'\n"
              f"Max value: {round(float(find_max(img)[1]),5)}\n"
              f"Location: {find_max(img)[0]}")
    x = center[0]
    y = center[1]
    ss1 = math.floor(k_sz[0]/2)
    ss2 = math.floor(k_sz[0] / 2)
    k1 = np.linspace(-ss1, ss1, int(k_sz[0]))
    k2 = np.linspace(-ss2, ss2, int(k_sz[1]))
    k_x, k_y = np.meshgrid(k1, k2)
    i = 0
    while True:
        i += 1
        if i >= n_iters:
            if display:
                print(f"\nFound:\n"
                      f"Value: {round(float(img[int(y), int(x)]), 5)}\n"
                      f"Location: ({int(y)}, {int(x)})\n"
                      f"Iterations: {i}")

            return int(x), int(y)
        patch = get_patch(img, (x, y), k_sz)[0]
        x_k1 = np.divide(np.sum(np.multiply(k_x, patch)), np.sum(patch))
        y_k1 = np.divide(np.sum(np.multiply(k_y, patch)), np.sum(patch))

        if x_k1 < thresh and y_k1 < thresh:
            if display:
                print(f"\nFound:\n"
                      f"Value: {round(float(img[int(y), int(x)]), 5)}\n"
                      f"Location: ({int(y)}, {int(x)})\n"
                      f"Iterations: {i}")

            return int(x), int(y)
        else:
            x += x_k1
            y += y_k1


if __name__ == '__main__':
    ''' 
        img : Input image                   DEFAULT: k_sz = generate_responses_1() 
 k_sz (w,h) : Kernel size (w x h)           DEFAULT: k_sz = (5,5) 
     center : Starting point                DEFAULT: center = (50,50) 
     thresh : Threshold to stop iteration   DEFAULT: thresh = 0.02 
    n_iters : Maximum number of iterations  DEFAULT: n_iters = 10000
    display : Display the results           DEFAULT: display = False
    '''
    MS_modeSeeking(img=generate_responses_2(100,100),
                   display=True,
                   center=(30, 40),
                   thresh=0.02,
                   k_sz=(20,20) )
