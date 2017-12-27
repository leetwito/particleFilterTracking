import pdb
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_mse as mse


# def ssim(src,dst,multichannel=False):
#     print mse(src, dst)


IMG_PATH = "Images/{index}.png"
FRAME_COUNT = 112

NUM_PARTICLES = 100
NUM_OF_STATE = 4


WINDOW_WIDTH = 40
WINDOW_HEIGHT = 110

S_X_COOR = 0
S_Y_COOR = 1
S_X_VELOCITY = 2
S_Y_VELOCITY = 3



def add_noise(mat):
    return mat + np.random.normal(0, 0.2, mat.shape[0]*mat.shape[1]).reshape(mat.shape)

def cut_sub_portion(mat, state):
    x_start_index = int(state[S_X_COOR]) - WINDOW_WIDTH/2
    x_end_index = int(state[S_X_COOR]) + WINDOW_WIDTH/2

    y_start_index = int(state[S_Y_COOR]) - WINDOW_HEIGHT/2
    y_end_index = int(state[S_Y_COOR]) + WINDOW_HEIGHT/2

    # print "X:", x_start_index, x_end_index, "Y:", y_start_index,y_end_index, "IMG SIZE", img.shape
    return img[y_start_index:y_end_index, x_start_index:x_end_index], (x_start_index, y_start_index), (x_end_index, y_end_index)


s_init = [280,139,0,0]  # todo change this

S = np.transpose(np.ones((NUM_PARTICLES,NUM_OF_STATE)) * s_init)
# S = add_noise(S)

for i in range(FRAME_COUNT):
    # load first image

    img = cv2.imread(IMG_PATH.format(index = str(i+1).zfill(3)))
    img_sub, (x_start_index, y_start_index), (x_end_index, y_end_index) = cut_sub_portion(img, s_init)
    if i == 0:
        org_sub = img_sub
    w = []  # weighs / distances
    # print img_sub.shape
    for i in range(NUM_PARTICLES):
        noisy_sub, (x_s, y_s), (x_e, y_e) = cut_sub_portion(img, S[:,i])
        cv2.rectangle(img, (x_s, y_s), (x_e, y_e), (0,255,0), 2)
        # w.append(ssim(org_sub, noisy_sub, multichannel=True)) # org <=> img
        w.append(ssim(img_sub, noisy_sub, multichannel=True)) # org <=> img

    w = np.array(w)
    w = np.square(w)
    w = w / np.sum(w)  # normalize
    c = np.cumsum(w)  # cumulative weights

    # choose and initialize the next step
    # i_max = np.argmax(w)
    # s_init = S[:,i_max]  # add S_init velocity
    s_init = np.dot(S, w)
    s_init[S_X_COOR:S_Y_COOR+1] += s_init[S_X_VELOCITY:S_Y_VELOCITY+1]  # add the velocity


    # choose the particle points for the next step
    # the chance to pick a point is proportional to it's weight - since we pick from cumulative weights according to uniform distribution
    S_next = np.zeros(S.shape)
    for i in range(NUM_PARTICLES):
        uniform_random = np.random.uniform()
        min_val = min(val for val in c if val > uniform_random)
        j = np.where(c == min_val)[0][0]
        S_next[:,i] = S[:,j]

    # add the velocity
    S_next = add_noise(S_next)
    S_next[S_X_COOR:S_Y_COOR+1,:] += S_next[S_X_VELOCITY:S_Y_VELOCITY+1,:]  # S_next[0:2,:] += S_next[2:4,:]

    # add noise to take area around repeating points
    S = S_next

    cv2.rectangle(img, (x_start_index, y_start_index), (x_end_index, y_end_index), (0,0,255), 2)
    cv2.imshow("df", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

