from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.metrics import check_scoring, accuracy_score, log_loss, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
#Authors:
#Juan Alejandro Ramon Ortega (s6470173)
#Erika Saiz (s5880734)
#Dániel Dávid (s6415997)

wine = load_wine()
wine_data_train, wine_data_test, wine_target_train, wine_target_test = train_test_split(wine.data, wine.target,
                                                                                        test_size=0.2, random_state=100)

def do_neural_network(train_data, test_data, is_grid_search: bool = False, is_cv: bool = False, is_loss: bool = False, is_matrix: bool = False, hidden_layers: tuple[int] = (10, ),
                       grid_search_criteria: str = "f1_weighted", solver_method: str = 'sgd', e_stopping:bool = True, mom: float = 0.5,
                         learning_rate_type: str = 'adaptive', learning_rate_value: float = 1, n_iter: int = 200):
    #setting up data
    scaler = StandardScaler()
    train_data, test_data = scaler.fit_transform(train_data), scaler.fit_transform(test_data) #train and test data seperated, no leakage
    mlp_classifier = MLPClassifier()

    if is_grid_search:
        gridsearch = GridSearchCV(mlp_classifier, {'learning_rate_init': [0.001, 0.1, 1, 2], 'momentum' : [0.5, 0.9, 0.99, 0.999],
                                                    'learning_rate':['adaptive', 'invscaling'], 'early_stopping':[True],
                                                      'random_state':[100], 'hidden_layer_sizes' : [(10,),(10,10), (20,20)], 'solver':['sgd']}, scoring=grid_search_criteria)
        gridsearch = gridsearch.fit(train_data, wine_target_train)
        mlp_classifier = gridsearch.best_estimator_
        print(mlp_classifier)
    else:
        if e_stopping:
            train_losses = []
            validation_losses = []
            TOLERANCE = 0.0001
            N_ITER_NO_CHANGE = 10
            n_not_improved = 0
            curr_train_data, val_data, train_target, val_target = train_test_split(train_data, wine_target_train, test_size= 0.1, random_state= 100)
            mlp_classifier = MLPClassifier(hidden_layer_sizes= hidden_layers, random_state= 100, solver= solver_method, max_iter= 1, warm_start= True,
                                                    early_stopping= False, momentum= mom, learning_rate= learning_rate_type, learning_rate_init= learning_rate_value)
            for _ in range(200):
                mlp_classifier = mlp_classifier.fit(curr_train_data, train_target)
                train_losses.append(mlp_classifier.loss_)
                predictions_val = mlp_classifier.predict_proba(val_data)
                cur_val_loss = log_loss(val_target, predictions_val)

                if len(validation_losses) > 0 and cur_val_loss >= validation_losses[-1] - TOLERANCE: #if loss doesn't improve
                    n_not_improved += 1
                else:
                    n_not_improved = 0
                validation_losses.append(cur_val_loss)

                if n_not_improved > N_ITER_NO_CHANGE:
                    break
            
            print(len(validation_losses))
            print(len(train_losses))
            print_loss_curve(train_losses, validation_losses)
        else:  
            mlp_classifier = MLPClassifier(hidden_layer_sizes= hidden_layers, random_state= 100, solver= solver_method, 
                                        early_stopping= e_stopping, momentum= mom, learning_rate= learning_rate_type, learning_rate_init= learning_rate_value).fit(train_data, wine_target_train)
            print(mlp_classifier)
            print(len(mlp_classifier.loss_curve_))

    if is_cv:
        cv_scores = cross_val_score(mlp_classifier, train_data, wine_target_train, cv = 5, scoring= 'f1_weighted')
        return np.mean(cv_scores), np.std(cv_scores)
    if is_loss:
        predictions_train_proba = mlp_classifier.predict_proba(train_data)
        predictions_test_proba = mlp_classifier.predict_proba(test_data)
        train_loss = log_loss(wine_target_train, predictions_train_proba)
        test_loss = log_loss(wine_target_test, predictions_test_proba)
       

   # predictions
    predictions_train = mlp_classifier.predict(train_data)
    predictions_test = mlp_classifier.predict(test_data)

    # accuracy
    train_accuracy = accuracy_score(wine_target_train, predictions_train)
    test_accuracy = accuracy_score(wine_target_test, predictions_test)

    if is_matrix:
        train_matrix = confusion_matrix(wine_target_train, predictions_train, labels=[0, 1, 2])
        test_matrix = confusion_matrix(wine_target_test, predictions_test, labels=[0, 1, 2])
        print("Train confusion matrix")
        print(train_matrix)
        print("Test confusion matrix")
        print(test_matrix)
    # F1 Score
    train_f1_score = f1_score(wine_target_train, predictions_train, average="weighted")
    test_f1_score = f1_score(wine_target_test, predictions_test, average="weighted")


    if is_loss:
        return train_accuracy, test_accuracy, train_loss, test_loss, train_f1_score, test_f1_score

    return train_accuracy, test_accuracy, train_f1_score, test_f1_score

def print_loss_curve(train_losses, validation_losses):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sns.lineplot(x = [i for i in range(len(train_losses))], y = train_losses, ax= ax1, color = "blue", label = "Train losses")
    sns.lineplot(x = [i for i in range(len(train_losses))], y = validation_losses, ax= ax2, color = "green", label = "Validation losses")
    ax2.set_ylim(0,3)
    ax1.set_xlabel("Number of epochs (n)")
    ax1.set_ylabel("Log loss")
    plt.legend()
    ax1.legend(loc='upper left')
    plt.title("Loss plot")
    plt.show()

print(do_neural_network(wine_data_train, wine_data_test, e_stopping= True, is_loss= True))
#result = do_neural_network(wine_data_train, wine_data_test, is_loss= True, is_matrix= True, hidden_layers=(50, 50))
#print(f"Training accuracy: {result[0]}, testing accuracy: {result[1]}, train log loss: {result[2]}, test log loss: {result[3]}, train F1 score: {result[4]}, test F1 score: {result[5]}")



'''MLPClassifier(early_stopping=True, hidden_layer_sizes=(10,),
              learning_rate='adaptive', learning_rate_init=1, momentum=0.5,
              random_state=100, solver='sgd')
              (0.9929577464788732, 0.9722222222222222, 0.18004186304553632, 0.2981836523454897, 0.9929756179508213, 0.9725602884832093)'''