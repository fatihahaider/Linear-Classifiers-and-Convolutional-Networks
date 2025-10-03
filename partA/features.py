import numpy as np
import cv2
import matplotlib.pyplot as plt


############## SIFT ############################################################

def extract_sift(img, step_size=1):
    """
    Extract SIFT features for a given grayscale image. Instead of detecting 
    keypoints, we will set the keypoints to be uniformly distanced pixels.
    Feel free to use OpenCV functions.
    
    Note: Check sift.compute and cv2.KeyPoint

    Args:
        img: Grayscale image of shape (H, W)
        step_size: Size of the step between keypoints.
        
    Return:
        descriptors: numpy array of shape (int(img.shape[0]/step_size) * int(img.shape[1]/step_size), 128)
                     contains sift feature.
    """
    sift = cv2.SIFT_create() # or cv2.xfeatures2d.SIFT_create()
    descriptors = np.zeros((int(img.shape[0]/step_size) * int(img.shape[1]/step_size), 128))

    
    ### START YOUR CODE HERE ###
    keypoints = [] # build grid of keypoints
    for y in range(0, img.shape[0], step_size):
        for x in range(0, img.shape[1], step_size):
            keypoint = cv2.KeyPoint(x=float(x) , y=float(y) , size=step_size) # =float(x) 
            keypoints.append(keypoint)

    keypoints, descriptors = sift.compute(img, keypoints) # compute the D for each keypoint
    ### END YOUR CODE HERE ###

    return descriptors



def extract_sift_for_dataset(data, step_size=1):
    all_features = []
    for i in range(len(data)):
        img = data[i]
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        descriptors = extract_sift(img, step_size) # use function above to map keypoints and compute descriptors 
        all_features.append(descriptors)
    return np.stack(all_features, 0)



