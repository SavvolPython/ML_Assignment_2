from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Authors:
# Juan Alejandro Ramon Ortega (s6470173)
# Erika Saiz (s5880734)
# Dániel Dávid (s6415997)

wine = load_wine()
wine_data_train, wine_data_test, wine_target_train, wine_target_test = train_test_split(wine.data, wine.target,
                                                                                        test_size=0.2, random_state=100)


def mlp_grid_search(train_data, test_data, grid_search_criteria: str = "f1_weighted"):
    # setting up data
    scaler = StandardScaler()
    train_data, test_data = scaler.fit_transform(train_data), scaler.fit_transform(
        test_data)
    mlp_classifier = MLPClassifier()
    gridsearch = GridSearchCV(mlp_classifier, {'learning_rate_init': [0.001, 0.1, 1, 2], 'momentum': [0.5, 0.9, 0.99, 0.999],
                                               'learning_rate': ['adaptive', 'invscaling'], 'early_stopping': [True],
                                               'random_state': [100], 'hidden_layer_sizes': [(10,), (10, 10), (20, )], 'solver': ['sgd']}, scoring=grid_search_criteria)
    gridsearch = gridsearch.fit(train_data, wine_target_train)
    mlp_classifier = gridsearch.best_estimator_

    return mlp_classifier


def do_neural_network(train_data, test_data, train_target, test_target, is_cv: bool = False, is_matrix: bool = False, hidden_layer_sizes: tuple[int] = (20, ), early_stopping: bool = True):
    # setting up data
    scaler = StandardScaler()
    train_data, test_data = scaler.fit_transform(train_data), scaler.fit_transform(
        test_data)  
    mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=100, solver='sgd', early_stopping=early_stopping,
                                       learning_rate='adaptive', learning_rate_init=1, momentum=0.5).fit(train_data, train_target)

    if is_cv:
        cv_scores = cross_val_score(
            mlp_classifier, train_data, train_target, cv=5, scoring='f1_weighted')
        return np.mean(cv_scores), np.std(cv_scores)

   # predictions
    predictions_train = mlp_classifier.predict(train_data)
    predictions_test = mlp_classifier.predict(test_data)
    predictions_train_proba = mlp_classifier.predict_proba(train_data)
    predictions_test_proba = mlp_classifier.predict_proba(test_data)

    # accuracy
    train_accuracy = accuracy_score(train_target, predictions_train)
    test_accuracy = accuracy_score(test_target, predictions_test)

    # F1 Score
    train_f1_score = f1_score(
        train_target, predictions_train, average="weighted")
    test_f1_score = f1_score(
        test_target, predictions_test, average="weighted")

    # losses
    train_loss = log_loss(train_target, predictions_train_proba)
    test_loss = log_loss(test_target, predictions_test_proba)

    if is_matrix:
        train_matrix = confusion_matrix(
            train_target, predictions_train, labels=[0, 1, 2])
        test_matrix = confusion_matrix(
            test_target, predictions_test, labels=[0, 1, 2])
        print("Train confusion matrix")
        print(train_matrix)
        print("Test confusion matrix")
        print(test_matrix)

    return train_accuracy, test_accuracy, train_loss, test_loss, train_f1_score, test_f1_score


def do_loss_plot(data, target, hidden_layer_sizes, n_iter: int = 200):
    # setting up data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    train_losses = []
    validation_losses = []
    curr_train_data, val_data, curr_train_target, val_target = train_test_split(
        data, target, test_size=0.1, random_state=100)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=100, solver='sgd',
                                   max_iter=1, warm_start=True, learning_rate='adaptive', learning_rate_init=1, momentum=0.5)
    for _ in range(n_iter):
        mlp_classifier = mlp_classifier.partial_fit(
            curr_train_data, curr_train_target, classes=np.unique(curr_train_target))
        train_losses.append(mlp_classifier.loss_)
        predictions_val = mlp_classifier.predict_proba(val_data)    
        cur_val_loss = log_loss(val_target, predictions_val)
        validation_losses.append(cur_val_loss)

    print_loss_curve(train_losses, validation_losses, hidden_layer_sizes)


def print_loss_curve(train_losses, validation_losses, hidden_layer_sizes):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sns.lineplot(x=[i for i in range(len(train_losses))],
                 y=train_losses, ax=ax1, color="blue", label="Train losses")
    sns.lineplot(x=[i for i in range(len(train_losses))], y=validation_losses,
                 ax=ax2, color="green", label="Validation losses")
    ax2.set_ylim(0, 3)
    ax1.set_xlabel("Number of epochs (n)")
    ax1.set_ylabel("Log loss")
    plt.legend()
    ax1.legend(loc='upper left')
    plt.title(f"Loss plot for {hidden_layer_sizes} architecture")
    plt.show()


result = do_neural_network(wine_data_train, wine_data_test, wine_target_train,
    wine_target_test, is_matrix=False, hidden_layer_sizes=(10,), early_stopping=True)
print(f"Training accuracy: {result[0]}, testing accuracy: {result[1]}, train log loss: {result[2]}, test log loss: {result[3]}, train F1 score: {result[4]}, test F1 score: {result[5]}")



