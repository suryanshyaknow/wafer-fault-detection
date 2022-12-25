from wafer.logger import lg
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from dataclasses import dataclass


@dataclass
class BestModelSelection:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.BestModelSelection" class')

    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array

    xgb_clf = XGBClassifier(objective='binary:logistics')
    random_clf = RandomForestClassifier(random_state=42)
    svc_clf = SVC(kernel='rbf')

    def choose_best_candidate_model(self) -> str:
        try:
            lg.info("Quest for choosing the `best candidate model` begins..")
            lg.info('Candidate models: "SVC", "RandomForestClassifier", "XGBClassifier"')

            if len(np.unique(self.y_train)) == 0:  # then can't use `roc_auc_score`, Will go ahead with `accuracy`

                ####################### Evaluation using Cross-Validation #########################
                lg.info('Evaluating the `SVC` using cross-validation..')
                cross_eval_scores = []
                # Evaluating the SVC(kernel=`rbf`)
                svc_mean_mean_score = cross_val_score(
                    self.svc_clf, self.X_train, self.y_train, scoring='accuracy', cv=5, verbose=2).mean()
                cross_eval_scores.append((svc_mean_mean_score, "SVC"))
                lg.info("..cross-validaiton finished successfully!")
                lg.info(f"Mean `SVC` score: {svc_mean_mean_score}")

                # Evaluating the Random Forest Clf
                lg.info('Evaluating the `RandomForestClassifier` using cross-validation..')
                random_clf_mean_score = cross_val_score(
                    self.random_clf, self.X_train, self.y_train, scoring='accuracy', cv=5, verbose=2).mean()
                cross_eval_scores.append((random_clf_mean_score, "RandomForestClassifier"))
                lg.info("..cross-validaiton finished successfully!")
                lg.info(f"Mean `RandomForestClassifier` score: {random_clf_mean_score}")

                # Evaluating the XGB Clf
                lg.info('Evaluating the `XGBClassifier` using cross-validation..')
                xgb_clf_mean_score = cross_val_score(
                    self.xgb_clf, self.X_train, self.y_train, scoring='accuracy', cv=5, verbose=2).mean()
                cross_eval_scores.append((xgb_clf_mean_score, "XGBClassifier"))
                lg.info("..cross-validaiton finished successfully!")
                lg.info(f"Mean `XGBClassifier` score: {xgb_clf_mean_score}")

                ##################### Returning the best performing model #########################
                cross_eval_scores.sort()
                best_candidate = cross_eval_scores[-1]  # with largest cross-validated accuracy

                lg.info(f'best performing classifier turned outta be "{best_candidate[1]}" with {best_candidate[0]}" accuracy!')
                return best_candidate[1]

            else:  # gonna go ahead with `roc_auc_score` as performance metric

                ####################### Evaluation using Cross-Validation #########################
                cross_eval_scores = []
                # Evaluating SVC(kernel=`rbf`)
                lg.info('Evaluating the `SVC` using cross-validation..')
                svc_mean_mean_score = cross_val_score(
                    self.svc_clf, self.X_train, self.y_train, scoring='roc-auc', cv=5, verbose=2).mean()
                cross_eval_scores.append((svc_mean_mean_score, "SVC"))
                lg.info("..cross-validaiton finished successfully!")
                lg.info(f"Mean `SVC` score: {svc_mean_mean_score}")

                # Evaluating Random Forest Clf
                lg.info('Evaluating the `RandomForestClassifier` using cross-validation..')
                random_clf_mean_score = cross_val_score(
                    self.random_clf, self.X_train, self.y_train, scoring='roc_auc', cv=5, verbose=2).mean()
                cross_eval_scores.append((random_clf_mean_score, "RandomForestClassifier"))
                lg.info("..cross-validaiton finished successfully!")
                lg.info(f"Mean `RandomForestClassifier` score: {random_clf_mean_score}")

                # Evaluating XGB Clf
                lg.info('Evaluating the `XGBClassifier` using cross-validation..')
                xgb_clf_mean_score = cross_val_score(
                    self.xgb_clf, self.X_train, self.y_train, scoring='roc_auc', cv=5, verbose=2).mean()
                cross_eval_scores.append((xgb_clf_mean_score, "XGBClassifier"))
                lg.info("..cross-validaiton finished successfully!")
                lg.info(f"Mean `XGBClassifier` score: {xgb_clf_mean_score}")

                ##################### Returning the best performing model #########################
                cross_eval_scores.sort()
                best_candidate = cross_eval_scores[-1]  # model with largest cross-validated accuracy
                
                lg.info(f'best performing classifier turned outta be "{best_candidate[1]}" with {best_candidate[0]}" ROC-AUC score!')
                return best_candidate[1]

        except Exception as e:
            lg.exception(e)
            raise e

    def hypertune_XGBClassifier(self) -> XGBClassifier:
        try:
            lg.info("Hypertuning the `XGBClassifier` using GridSearchCV..")

            # shortlist params for GridSearchCV
            grid_params = {
                'learning_rate': [.001, .01, .1, .5],
                'max_depth': [3, 5, 6],
                'n_estimators': [100, 300, 500, 1000]
            }
            # commence GridSearchCV
            self.grid_search = GridSearchCV(self.xgb_clf, param_grid=grid_params, cv=5, verbose=2)
            self.grid_search.fit(self.X_train, self.y_train)
            lg.info(f"..GridSearchCV finished successfully with best_score_={self.grid_search.best_score_}!")
            lg.info(f"XGBClassifier's best params: {self.grid_search.best_params_}")

            lg.info("Returning the `XGBClassifier` trained using best_params_..") 
            return self.grid_search.best_estimator_
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def hypertune_RandomForestClassifier(self) -> RandomForestClassifier:
        try:
            lg.info("Hypertuning the `RandomForestClassifier` using GridSearchCV..")

            # shortlist params for GridSearchCV
            grid_params = {
                "n_estimators": [100, 300, 500, 1000], 
                "criterion": ['gini', 'entropy'], 
                "max_depth": [3, 5, 6], 
                "max_features": ['auto', 'log2']
            }
            # commmence GridSearchCV
            self.grid_search = GridSearchCV(self.random_clf, param_grid=grid_params, cv=5, verbose=2)
            self.grid_search.fit(self.X_train, self.y_train)
            lg.info(f"..GridSearchCV finished successfully with best_score_={self.grid_search.best_score_}!")
            lg.info(f"RandomForestClassifier's best params: {self.grid_search.best_params_}")

            lg.info("Returning the `RandomForestClassifier` trained using best_params_..")
            return self.grid_search.best_estimator_
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def hypertune_SVC(self) -> SVC:
        try:
            lg.info("Hypertuning the `SVC` using GridSearchCV..")

            # shortlist params for GridSearchCV
            grid_params = {
                "gamma": [0.1, 1, 10],
                "C": [.1, 1, 10]
            }

            # commmence GridSearchCV
            self.grid_search = GridSearchCV(self.svc_clf, param_grid=grid_params, cv=5, verbose=2)
            self.grid_search.fit(self.X_train, self.y_train)
            lg.info(f"..GridSearchCV finished successfully with best_score_={self.grid_search.best_score_}!")
            lg.info(f"SVC's best params: {self.grid_search.best_params_}")

            lg.info("Returning the `SVC` trained using best_params_..")
            return self.grid_search.best_estimator_
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def get_best_model(self):
        try:

            ################################## Fetch Best Model #################################################
            lg.info("Fetching the best model evaluated using cross-vaidation..")
            best_candidate = self.choose_best_candidate_model()
            lg.info(f'Best Model we\'ve got: "{best_candidate}"')

            ################################# Finetune Best Model ###############################################
            lg.info(f'finetuning the best model `{best_candidate}`..')
            if best_candidate == "SVC":
                best_mod = self.finetune_SVC()
            elif best_candidate == "RandomForestClassifier":
                best_mod = self.finetune_RandomForestClassifier()
            else:
                best_mod = self.finetune_XGBClassifier()
            lg.info("..finetuning of the best model completed with success!")
            
            ############################### Evaluate Best Model #################################################
            if len(np.unique(self.y_train)) == 0:  # performance metric: `accuracy`
                # performance on Training set
                lg.info(f"evaluating the performance of `{best_candidate}` on training set..")
                train_acc =  best_mod.score(self.X_train, self.y_train)
                lg.info(f"Accuracy on training set: {train_acc}")
                # performance on Test set
                lg.info(f"evaluating the true performance of `{best_candidate}` on test set..")
                test_acc =  best_mod.score(self.X_train, self.y_train)
                lg.info(f"Accuracy on training set: {test_acc}")
                ...
            else:  # performanve metric: `roc_auc_score`
                # performance on Training set
                lg.info(f"evaluating the performance of `{best_candidate}` on training set..")
                train_acc =  best_mod.score(self.X_train, self.y_train)
                lg.info(f"AUC on training set: {train_acc}")
                # performance on Test set
                lg.info(f"evaluating the true performance of `{best_candidate}` on test set..")
                test_acc =  best_mod.score(self.X_train, self.y_train)
                lg.info(f"AUC on training set: {test_acc}")
            
            lg.info(f"returning the best model `{best_candidate}`..")
            return best_candidate, best_mod
            ...
        except Exception as e:
            lg.exception(e)
            raise e
