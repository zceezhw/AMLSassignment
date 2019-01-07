from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import csv
import landmarks as l2
import numpy as np


def get_data():

    # use function from landmark.py to extract features and labels

    X, y = l2.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T

    # Use 80% of the dataset as training dataset, and remaining 20% as testing datase

    tr_X = X[:3563]
    tr_Y = Y[:3563]
    te_X = X[3563:]
    te_Y = Y[3563:]

    return tr_X, tr_Y, te_X, te_Y


def train_SVM(training_images, training_labels, test_images, test_labels):
    sizeTr = len(training_images)
    sizeTe = len(test_images)
    TD_train = training_images.reshape(sizeTr, -1)
    TD_test = test_images.reshape(sizeTe, -1)

    # fit the SVM with training data

    svclassifier = svm.SVC(C=1, kernel='linear', class_weight='balanced', random_state=0)
    svclassifier.fit(TD_train, training_labels[:, 0])

    # make prediction using unseen dataset

    y_pred = svclassifier.predict(TD_test)

    # use to generate CSV file with the predicted label

    with open('Task_1.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(y_pred))

    # print confusion matrix for evaluation

    print(confusion_matrix(test_labels[:, 0], y_pred))
    print(classification_report(test_labels[:, 0], y_pred))


if __name__ == '__main__':
    tr_X, tr_Y, te_X, te_Y = get_data()
    train_SVM(tr_X, tr_Y, te_X, te_Y)

