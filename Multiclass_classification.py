
import pandas as pd
import numpy as np
from PIL import Image
import os, os.path
from sklearn.model_selection import train_test_split, GridSearchCV
import keras


All_Images = np.zeros((5000, 256, 256, 1))
X_total = np.zeros((4454, 256, 256, 1))

valid_images = ['.png', '.PNG']
for imagename in os.listdir('/Users/bill/PycharmProjects/LAB2/dataset/dataset_t'):
    ind = int(str(os.path.splitext(imagename)[0]))
    ext = os.path.splitext(imagename)[1]
    if ext.lower() not in valid_images:
        continue
    img = Image.open(os.path.join('/Users/bill/PycharmProjects/LAB2/dataset/dataset_t', imagename)).convert('L')
    All_Images[ind] = np.array(img.getdata(), dtype=np.uint8).reshape(256, 256, 1)

count = 0
for i in range(5000):
    if sum(sum(All_Images[i])) != 0:
        X_total[count] = All_Images[i]
        count = count + 1

labels = pd.read_csv('/Users/bill/PycharmProjects/LAB2/dataset/attribute_list_face.csv')

Y_total = np.array(labels['hair_color'])

# Convert the label class into a one-hot representation

num_classes = 6

row_delete = []
for i in range(4454):
    if Y_total[i] == -1:
        row_delete.append(i)

X_final = np.delete(X_total, row_delete, axis=0)
Y_final = np.delete(Y_total, row_delete, axis=0)

x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=0.2, shuffle=False)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def train_CNN(training_images, training_labels, test_images, test_labels):

    # because images are 256*256

    training_images = training_images / 255.0
    training_labels = training_labels
    test_images = test_images / 255.0
    test_labels = test_labels

    model = keras.models.Sequential()

    # add model layers

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(6, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.0007, decay=0.0, momentum=0.0, nesterov=False)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_images, training_labels, batch_size=128, epochs=50)

    res = model.evaluate(x=test_images, y=test_labels, batch_size=128, verbose=1, sample_weight=None, steps=None)
    predictions = model.predict(x_test, batch_size=128, verbose=0, steps=None)
    print('The classification accuracy on the test set is:', res[1])

    return 0


train_CNN(x_train, y_train, x_test, y_test)
