import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

## IOU in pure numpy
def numpy_iou(y_true, y_pred, n_class=2):
    # IOU = TP/(TP+FN+FP)

    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))

    IOU = []

    for c in range(n_class):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        n = TP
        d = float(TP + FP + FN + 1e-10)

        iou = np.divide(n, d)
        IOU.append(iou)

    return np.mean(IOU)

## Calculating IOU across a range of thresholds, then we will mean all the
## values of IOU's.
## this function can be used as keras metrics
def numpy_mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.01):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.numpy_function(numpy_iou, [y_true, y_pred_], tf.float64)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

if __name__ == "__main__":
    ## Seeding
    tf.compat.v1.random.set_random_seed(1234)

    ## Defining Placeholders
    shape = [8, 256, 256, 1]
    y_true = tf.placeholder(tf.float32, shape=shape)
    y_pred = tf.placeholder(tf.float32, shape=shape)

    ## Reading the masks from the path
    y_true_masks = np.zeros((8, 256, 256, 1))
    for idx, path in enumerate(os.listdir("ds/")):
        mask = cv2.imread("ds/" + path, -1)
        mask = cv2.resize(mask, (256, 256))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask/255.0
        y_true_masks[idx] = mask

    ## Calculating the predicting the masks
    ## We have used the true mask as the predicting mask, we have just shuffle
    ## the dataset and then flip the images.
    y_pred_masks = y_true_masks
    np.random.shuffle(y_pred_masks)
    y_pred_masks = np.flip(y_pred_masks)

    ## Session
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        ## Mean IOU
        miou = numpy_mean_iou(y_true, y_pred)
        miou = sess.run(miou, feed_dict={y_true: y_true_masks, y_pred: y_pred_masks})
        print("Numpy mIOU: ", miou)

        ## Preprocessing for saving the masks, and viewing them as images.
        yt = y_true_masks
        yp = y_pred_masks

        yt = yt*255
        yp = yp*255

        rows = 2
        cols = 8
        h = 256
        w = 256

        images = np.array([yt, yp])
        images = images.reshape((rows, cols, h, w, 1))
        images = images.transpose(0, 2, 1, 3, 4)
        images = images.reshape((rows * h, cols * w, 1))

        cv2.imwrite("mask.png", images)
