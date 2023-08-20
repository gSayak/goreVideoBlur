import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

checkpoint_path = "./InceptionV3_weights.h5"
InceptionV3_base = tf.keras.applications.InceptionV3(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")

for layer in InceptionV3_base.layers:
    layer.trainable = False

InceptionV3 = tf.keras.Sequential()
InceptionV3.add(InceptionV3_base)
InceptionV3.add(tf.keras.layers.Dropout(0.5))
InceptionV3.add(tf.keras.layers.Flatten())
InceptionV3.add(tf.keras.layers.BatchNormalization())
InceptionV3.add(tf.keras.layers.Dense(32, kernel_initializer='he_uniform'))
InceptionV3.add(tf.keras.layers.BatchNormalization())
InceptionV3.add(tf.keras.layers.Activation('relu'))
InceptionV3.add(tf.keras.layers.Dropout(0.5))
InceptionV3.add(tf.keras.layers.Dense(32, kernel_initializer='he_uniform'))
InceptionV3.add(tf.keras.layers.BatchNormalization())
InceptionV3.add(tf.keras.layers.Activation('relu'))
InceptionV3.add(tf.keras.layers.Dropout(0.5))
InceptionV3.add(tf.keras.layers.Dense(32, kernel_initializer='he_uniform'))
InceptionV3.add(tf.keras.layers.BatchNormalization())
InceptionV3.add(tf.keras.layers.Activation('relu'))
InceptionV3.add(tf.keras.layers.Dense(1, activation='sigmoid'))

InceptionV3.load_weights(checkpoint_path)

cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)  # load a video
cap.set(3, cam_width)
cap.set(4, cam_height)
while True:
    ret, frame = cap.read()  # reading video by frame

    img = cv2.resize(frame, (224, 224))  # resizing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # transform

    pred = InceptionV3.predict(img.reshape((1, 224, 224, 3)), verbose=0)
    print(pred[0][0])
    if pred[0][0] < 0.5:
        # kernel = np.ones((50, 50), np.float32)/100
        frame = cv2.blur(frame, (40, 40), cv2.BORDER_DEFAULT)

    cv2.imshow("YOLOv5 Object Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break
