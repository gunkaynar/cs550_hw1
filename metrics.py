from sklearn.metrics import confusion_matrix
#accuracy and confusion matrix
def scores(labels, predicted_labels):
    cm = confusion_matrix(labels, predicted_labels)

    return cm