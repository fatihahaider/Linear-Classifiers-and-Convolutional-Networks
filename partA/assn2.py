import numpy as np
import cv2


def load_average_color_with_bias(X_data):
    """get average color of each image. Add bias dimension at the end with value 1.
    
    Arguments:
        X_data: numpy array of size (N, H, W, 3)
    
    Outputs:
        output: numpy array of size (N, 4)
    """
    X_data = X_data.copy()
    N = X_data.shape[0]
    output = np.zeros([N, 4], dtype=X_data.dtype)
    
    ### START YOUR CODE HERE ###
    for i in range(N):
        vector = np.mean(X_data[i], axis = (0,1))
        vector = np.concatenate([vector, [1]])     
        output[i] = vector
    ### END YOUR CODE HERE ###
    
    return  output


def load_flatten(X_data):    
    """flatten the data.
    
    Arguments:
        X_data: numpy array of size (N, H * W, D)
        
    Outputs:
        output: numpy array of size (N * H * W, D)
    """
    X_data = X_data.copy()
    N, HW, D = X_data.shape
    
    ### START YOUR CODE HERE ###
    output = X_data.reshape(-1,D)
    ### END YOUR CODE HERE ###
    
    return  output

def load_histogram_with_bias(X_data, centroids):
    """given centroid, assign label to each of the keypoints. Draw Histogram
    
    Arguments:
        X_data: numpy array of size (N, P, D), where N is number of images,
                P is number of keypoints, and D is dimension of features
        centroids: numpy of array of size (K, D), where K is number of centroids.     
    
    Outputs:
        X_hist: numpy array of size (N, K+1), where X_hist[i,j] contains number of 
                keypoints from image i that is closest to centroid[j].
                X_hist[:, K] should be 1 for bias.
    """
    X_data, centroids = X_data.copy(), centroids.copy()
    N, P, D = X_data.shape # example would be 5 im, 10 kp, 128 features
    K, D = centroids.shape # K by D array 
    X_hist = np.zeros([N, K+1])
    
    
    ### START YOUR CODE HERE ###
    for i in range(N):
        for p in range(P): # for each keypoint
            diffs = centroids - X_data[i, p] # each keypoint in each image       
            dists = np.sum(diffs**2, axis=1) 
            j = np.argmin(dists)                     
            X_hist[i, j] += 1                        

        X_hist[:, K] = 1
    ### END YOUR CODE HERE ###
    
    return X_hist   
    
def load_vector_image_with_bias(X_train, X_val, X_test):
    """Reshape the image data into rows
       Normalize the data by subtracting the mean training image from all images.
       Add bias dimension and transform into columns
    
    Arguments:
        X_train: numpy array of size (N_train, H, W, 3), where N_train is number of images
        X_val: numpy array of size (N_val, H, W, 3), where N_val is number of images
        X_test: numpy array of size (N_text, H, W, 3), where N_text is number of images
    
    Outputs:
        X_train: numpy array of size (N, H * W * 3 + 1). Bias dimension at the end.
        X_val: numpy array of size (N, H * W * 3 + 1). Bias dimension at the end.
        X_test: numpy array of size (N, H * W * 3 + 1). Bias dimension at the end.
    """
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    N_train, N_val, N_text = X_train.shape[0], X_val.shape[0], X_test.shape[0]
    
    ### START YOUR CODE HERE ###
    X_train = X_train.reshape(N_train, -1)
    X_val = X_val.reshape(N_val, -1)
    X_test = X_test.reshape(N_text, -1)

    meanim = np.mean(X_train, 0)

    X_train -= meanim
    X_test -= meanim
    X_val -= meanim

    X_train = np.concatenate([X_train, np.ones((N_train, 1))], axis=1)
    X_val   = np.concatenate([X_val,   np.ones((N_val, 1))],   axis=1)
    X_test  = np.concatenate([X_test, np.ones((N_text, 1))], axis=1)
    ### END YOUR CODE HERE ###
    
    return X_train, X_val, X_test
    