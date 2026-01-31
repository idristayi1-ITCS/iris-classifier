# Iris-Classifier Project

# DATA PREPARATION -- Loading / importing the data

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data

y = iris.target

class_names =iris.target_names

print(iris.feature_names, iris.target_names)

# Splitting into training and test sets using a 80/20 split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random state of 42 for deterministic shuffling

# MODEL TRAINING -- Choosing and initialising the model - Decision Tree selected

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42) # Random state for reproducibility 

# Training the model using the training data

Modelled = model.fit(X_train, y_train)

# PREDICTION -- Making predictions using the training model

y_pred = Modelled.predict(X_test)

# Printing the first 5 predicted data vs the actual data

print("Predictions: ", y_pred[:5])
print("True labels: ", y_test[:5])

# EVALUATION -- Evaluating the Model using the accuracy and confusion matrix

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import numpy as np

titles_options = [("Confusion Matrix, without normalization", None), ("Normalized Confusion Matrix", "true")]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(Modelled, X_test, y_test, display_labels =class_names, cmap = plt.cm.Blues, normalize = normalize,)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()


# INTERPRETATION --From the accuracy score i.e. 100%, we see the model was able to predict correctly the species names of all the test data
# The Confuion matrix also revealed the model successfully predicted the specie names without false prediction for any specie.
