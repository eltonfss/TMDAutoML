from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_validate
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import itertools
import math
import autosklearn.classification
from sklearn import decomposition
from sklearn.decomposition import PCA
import sklearn.cross_validation as scv
import numpy as np
from time import time

from TMDataset import TMDataset
import const
import util


class TMDetection:
    dataset = TMDataset()
    classes = []
    classes2string = {}
    classes2number = {}

    def __init__(self, dataset=None):
        if dataset:
            self.dataset = dataset
        print("HAVE_DT", const.HAVE_DT)
        if not const.HAVE_DT:
            self.dataset.create_balanced_dataset(const.SINTETIC_FILE_TRAINING)
        classes_dataset = self.dataset.get_dataset['target'].values
        print(classes_dataset)
        for i, c in enumerate(sorted(set(classes_dataset))):
            self.classes2string[i] = c
            self.classes2number[c] = i
            self.classes.append(c)
            
    def _apply_pca(self, data_features, number_of_components=2):
        pca = PCA(n_components=number_of_components)
        pca.fit(data_features)
        data_features = pca.transform(data_features) 
        return data_features
        
    def _get_sets_for_classification(self, df_train, df_test, features, apply_pca=False, number_of_components=2):
        train, test = util.fill_nan_with_mean_training(df_train, df_test)
        train_features = train[features].values            
        train_classes = [self.classes2number[c] for c in train['target'].values]
        test_features = test[features].values
        test_classes = [self.classes2number[c] for c in test['target'].values]
        if apply_pca:
            train_features = self._apply_pca(train_features, number_of_components)
            test_features = self._apply_pca(test_features, number_of_components)
        return train_features, train_classes, test_features, test_classes  
    
    def _get_data_for_cross_validation(self, sensors_set, apply_pca=False, number_of_components=2):
        data = self.dataset.get_dataset
        data, useless = util.fill_nan_with_mean_training(data)
        data_features = data[sensors_set].values
        data_classes = [self.classes2number[c] for c in data['target'].values]
        if apply_pca:
            data_features = self._apply_pca(data_features, number_of_components)
        return data_features, data_classes
        
    def _calculate_metrics(self, y_true, y_predicted):
        acc = accuracy_score(y_true, y_predicted)
        report = classification_report(y_true, y_predicted)
        matrix = confusion_matrix(y_true, y_predicted)
        print("ACCURACY : " + str(acc))
        print("REPORT : " + str(report))
        print("CONFUSION MATRIX : " + str(matrix))
         
    def _evaluate_by_cross_validation(self, classifier, data_features, data_classes, num_folds=10):
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        scores = cross_validate(classifier, data_features, data_classes, scoring=scoring, cv=num_folds, return_train_score=False)
        mean_accuracy = scores['test_accuracy'].mean()
        mean_precision = scores['test_precision_macro'].mean()
        mean_recall = scores['test_recall_macro'].mean()
        mean_f1 = scores['test_f1_macro'].mean()
        mean_fit_time = scores['fit_time'].mean()
        mean_score_time = scores['score_time'].mean()
        print (num_folds, '-Fold Cross Validation Results')
        print ('MEAN ACCURACY:', mean_accuracy)
        print ('MEAN PRECISION:', mean_precision)
        print ('MEAN RECALL:', mean_recall)
        print ('MEAN F-MEASURE:', mean_f1)
        print ('MEAN FIT TIME:', mean_fit_time)
        print ('MEAN SCORE TIME:', mean_score_time)
        
    def _evaluate_by_stratified_cross_validation(self, classifier, data_features, data_classes, num_folds=10):
        
        # generate stratified samples
        cv_iter = scv.StratifiedKFold(data_classes, n_folds = num_folds, shuffle = True, random_state = 123)
        
        # initialize metric arrays
        accuracy_array = np.empty(num_folds)
        precision_array = np.empty(num_folds)
        recall_array = np.empty(num_folds)
        f1_array = np.empty(num_folds)
        fit_time_array = np.empty(num_folds)
        score_time_array = np.empty(num_folds)
        
        X = np.array(data_features)
        Y = np.array(data_classes)
        
        # perform cross-validation
        fold_counter = 0
        for train, test in cv_iter:
            X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]  
            t0=time()
            classifier.refit(X_train, y_train)
            t1=time()
            predictions = classifier.predict(X_test)
            accuracy_array[fold_counter] = accuracy_score(y_test, predictions)
            precision_array[fold_counter] = precision_score(y_test, predictions, average='macro') 
            recall_array[fold_counter] = recall_score(y_test, predictions, average='macro')
            f1_array[fold_counter] = f1_score(y_test, predictions, average='macro') 
            t2=time()
            fit_time_array[fold_counter] = round(t1-t0, 3)
            score_time_array[fold_counter] = round(t2-t1, 3)
            fold_counter += 1
        
        # print mean metrics
        print (num_folds, '-Fold Cross Validation Results')
        print ('MEAN ACCURACY:',  np.mean(accuracy_array))
        print ('MEAN PRECISION:', np.mean(precision_array))
        print ('MEAN RECALL:', np.mean(recall_array))
        print ('MEAN F-MEASURE:', np.mean(f1_array))
        print ('MEAN FIT TIME:', np.mean(fit_time_array))
        print ('MEAN SCORE TIME:', np.mean(score_time_array))
            
    def decision_tree(self, sensors_set, apply_pca=False, number_of_components=2):
        
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("DECISION TREE.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))

        # test set evaluation
        classifier_decision_tree = tree.DecisionTreeClassifier()
        t0=time()
        train_features, train_classes, test_features, test_classes = self._get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features, apply_pca, number_of_components
        )
        t1=time()
        classifier_decision_tree.fit(train_features, train_classes)
        t2=time()
        test_prediction = classifier_decision_tree.predict(test_features)
        self._calculate_metrics(test_classes, test_prediction)
        t3=time()
        print("SPLIT TIME:", round(t1-t0, 3))
        print("FIT TIME:", round(t2-t1, 3))
        print("SCORE TIME:", round(t3-t2, 3))
        
        # 10-fold cross-validation
        classifier_decision_tree = tree.DecisionTreeClassifier()
        t4=time()
        data_features, data_classes = self._get_data_for_cross_validation(features, apply_pca, number_of_components)
        t5=time()
        print("SPLIT TIME:", round(t5-t4, 3))
        self._evaluate_by_cross_validation(classifier_decision_tree, data_features, data_classes, 10)
        print("END TREE")

    def random_forest(self, sensors_set, apply_pca=False, number_of_components=2):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("RANDOM FOREST.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))
       
        # test set evaluation
        classifier_forest = RandomForestClassifier(n_estimators=const.PAR_RF_ESTIMATOR)
        t0=time()
        train_features, train_classes, test_features, test_classes = self._get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features, apply_pca, number_of_components
        )
        t1=time()
        classifier_forest.fit(train_features, train_classes)
        t2=time()
        test_prediction = classifier_forest.predict(test_features)
        self._calculate_metrics(test_classes, test_prediction)
        t3=time()
        print("SPLIT TIME:", round(t1-t0, 3))
        print("FIT TIME:", round(t2-t1, 3))
        print("SCORE TIME:", round(t3-t2, 3))
        
        
        # 10-fold cross-validation
        classifier_forest = RandomForestClassifier(n_estimators=const.PAR_RF_ESTIMATOR)
        t4=time()
        data_features, data_classes = self._get_data_for_cross_validation(features, apply_pca, number_of_components)
        t5=time()
        print("SPLIT TIME:", round(t5-t4, 3))
        self._evaluate_by_cross_validation(classifier_forest, data_features, data_classes, 10)
        print("END RANDOM FOREST")

    def neural_network(self, sensors_set, apply_pca=False, number_of_components=2):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("NEURAL NETWORK.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))
        
        # test set evaluation
        classifier_nn = MLPClassifier(hidden_layer_sizes=(const.PAR_NN_NEURONS[sensors_set],),
                                      alpha=const.PAR_NN_ALPHA[sensors_set], max_iter=const.PAR_NN_MAX_ITER,
                                      tol=const.PAR_NN_TOL)
        t0=time()
        train_features, train_classes, test_features, test_classes = self._get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features, apply_pca, number_of_components
        )
        t1=time()
        train_features_scaled, test_features_scaled = util.scale_features(train_features, test_features)
        t2=time()
        classifier_nn.fit(train_features_scaled, train_classes)
        t3=time()
        test_prediction = classifier_nn.predict(test_features_scaled)
        self._calculate_metrics(test_classes, test_prediction)
        t4=time()
        print("SPLIT TIME:", round(t1-t0, 3))
        print("SCALE TIME:", round(t2-t1, 3))
        print("FIT TIME:", round(t3-t2, 3))
        print("SCORE TIME:", round(t4-t3, 3))
        
        # 10-fold cross-validation
        classifier_nn = MLPClassifier(hidden_layer_sizes=(const.PAR_NN_NEURONS[sensors_set],),
                                      alpha=const.PAR_NN_ALPHA[sensors_set], max_iter=const.PAR_NN_MAX_ITER,
                                      tol=const.PAR_NN_TOL)
        t5=time()
        data_features, data_classes = self._get_data_for_cross_validation(features, apply_pca, number_of_components)
        t6=time()
        data_features_scaled, useless = util.scale_features(data_features)
        t7=time()
        print("SPLIT TIME:", round(t6-t5, 3))
        print("SCALE TIME:", round(t7-t6, 3))
        self._evaluate_by_cross_validation(classifier_nn, data_features_scaled, data_classes, 10)
        print("END NEURAL NETWORK")

    def support_vector_machine(self, sensors_set, apply_pca=False, number_of_components=2):
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("SUPPORT VECTOR MACHINE.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))
        
        # test set evaluation
        classifier_svm = SVC(C=const.PAR_SVM_C[sensors_set], gamma=const.PAR_SVM_GAMMA[sensors_set], verbose=False)
        t0=time()
        train_features, train_classes, test_features, test_classes = self._get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features, apply_pca, number_of_components
        )
        t1=time()
        train_features_scaled, test_features_scaled = util.scale_features(train_features, test_features)
        t2=time()
        classifier_svm.fit(train_features_scaled, train_classes)
        t3=time()
        test_prediction = classifier_svm.predict(test_features_scaled)
        self._calculate_metrics(test_classes, test_prediction)
        t4=time()
        print("SPLIT TIME:", round(t1-t0, 3))
        print("SCALE TIME:", round(t2-t1, 3))
        print("FIT TIME:", round(t3-t2, 3))
        print("SCORE TIME:", round(t4-t3, 3))
                
        # 10-fold cross-validation
        classifier_svm = SVC(C=const.PAR_SVM_C[sensors_set], gamma=const.PAR_SVM_GAMMA[sensors_set], verbose=False)
        t5=time()
        data_features, data_classes = self._get_data_for_cross_validation(features, apply_pca, number_of_components)
        t6=time()
        data_features_scaled, useless = util.scale_features(data_features)
        t7=time()
        print("SPLIT TIME:", round(t6-t5, 3))
        print("SCALE TIME:", round(t7-t6, 3))
        self._evaluate_by_cross_validation(classifier_svm, data_features_scaled, data_classes, 10)
        print("END SUPPORT VECTOR MACHINE.....")
        
    def auto_machine_learning(self, sensors_set, search_time=3600, apply_pca=False, number_of_components=2):
        
        features = list(self.dataset.get_sensors_set_features(sensors_set))
        print("AUTO MACHINE LEARNING.....")
        print("CLASSIFICATION BASED ON THESE SENSORS: ", self.dataset.get_remained_sensors(sensors_set))
        print("NUMBER OF FEATURES: ", len(features))

        # test set evaluation
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=search_time)
        t0=time()
        train_features, train_classes, test_features, test_classes = self._get_sets_for_classification(
            self.dataset.get_train, self.dataset.get_test, features, apply_pca, number_of_components
        )
        t1=time()
        automl.fit(train_features.copy(), train_classes.copy())
        t2=time()
        test_prediction = automl.predict(test_features)
        self._calculate_metrics(test_classes, test_prediction)
        t3=time()
        print("SPLIT TIME:", round(t1-t0, 3))
        print("FIT TIME:", round(t2-t1, 3))
        print("SCORE TIME:", round(t3-t2, 3))
        
        print("FINAL ENSEMBLE:", automl.show_models())
        
        # 10-fold cross-validation
        t4=time()
        data_features, data_classes = self._get_data_for_cross_validation(features, apply_pca, number_of_components)
        t5=time()
        print("SPLIT TIME:", round(t5-t4, 3))
        self._evaluate_by_stratified_cross_validation(automl, data_features, data_classes, 10)
        print("END AUTO MACHINE LEARNING")

