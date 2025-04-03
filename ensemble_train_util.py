from time import time

import numpy as np
from sklearn.base import BaseEstimator, clone
#from sklearn.model_selection import StratifiedKFold

from classification_util import ClassificationReport


class MyVotingClassifier(BaseEstimator):
    '''
    Similar to sklearn.VotingClassifier, but preserves the y data as a DataSeries.
    '''
    def __init__(self, estimators, voting='hard'):
        assert voting in ['hard', 'soft']
        self.estimators = [ (name, clone(model)) for name, model in estimators ]
        self.voting = voting

    def fit(self, X, y):
        for (name, estimator) in self.estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.voting == 'hard':
            preds = np.array([estimator.predict(X) for (name, estimator) in self.estimators])
            return np.array([np.argmax(np.bincount(preds[:, i])) for i in range(preds.shape[1])])
        else:
            pred_proba = np.array([estimator.predict_proba(X) for (name, estimator) in self.estimators])
            pred_proba = np.mean(pred_proba, axis=0)
            return np.array([np.argmax(p) for p in pred_proba])

    def predict_proba(self, X):
        pred_proba = np.array([estimator.predict_proba(X) for (name, estimator) in self.estimators])
        pred_proba = np.mean(pred_proba, axis=0) / len(self.estimators)
        return pred_proba
    

def extract_best_models_of_fold(results, fold):
    '''
    Seleciona todos os melhores modelos no fold informado (1 de cada tipo).
    '''
    best_models_of_fold = []
    for model_name in results.keys():
        best_model = clone(results[model_name]['best_models'][fold])
        best_model.set_params(**results[model_name]['best_parameters'][fold])
        best_models_of_fold.append((f"{model_name} #{fold+1}", best_model))
    return best_models_of_fold


def extract_all_best_models(results, n_folds):
    '''
    Seleciona os melhores modelos (de qualquer tipo) com seus parâmetros, sendo um para cada fold externo.
    Para cada fold externo, analisa o melhor de cada tipo de modelo (o melhor KNN, o melhor MLP, etc),  
    usando o f1-score e separa o melhor deles (que pode ser de qualquer tipo).
    Retorna a lista de modelos separados.
    '''
    best_models_per_fold = []
    for fold in range(n_folds):
        best_f1 = 0.0
        best_model = None
        best_model_name = None
        for model_name in results.keys():
            if results[model_name]['f1_score_list'][fold] > best_f1:
                best_f1 = results[model_name]['f1_score_list'][fold]
                best_model = clone( results[model_name]['best_models'][fold] )
                best_model.set_params(**results[model_name]['best_parameters'][fold])
                best_model_name = model_name
        best_models_per_fold.append((f"{best_model_name} #{fold+1}", best_model))
    return best_models_per_fold


def extract_best_models_of_type(results, model_name):
    '''
    Seleciona todas as configurações (hiperparametrizações) de um mesmo tipo de modelo, 
    sendo uma configuração para cada fold externo (onde ela foi a melhor configuração).
    '''
    return [ (f"{model_name} #{fold}", clone(model)) for fold, model in enumerate(results[model_name]['best_models']) ]


def train_ensemble(name_prefix, model_list, X, y, cv_outer):  
    """
    Train an ensemble model using cross-validation and store results using CVEvaluationResults.

    Args:
        model_list: List of tuples with model names and estimators.
        X: Feature dataset.
        y: Target labels.
        name_prefix: Prefix for naming the ensemble models.
        cv_outer: Cross-validation object (e.g., StratifiedKFold).

    Returns:
        dict: Results of the ensemble models.
    """
    # Create hard and soft voting classifiers
    _estimador_hard = MyVotingClassifier(estimators=model_list, voting='hard')
    _estimador_soft = MyVotingClassifier(estimators=model_list, voting='soft') 

    new_results = {}
    
    # Iterate over both hard and soft voting classifiers
    for name, estimador in [(name_prefix + '-hard-vote', _estimador_hard), (name_prefix + '-soft-vote', _estimador_soft)]:
        # Calculates and stores the results for this ensemble model
        model_results = ClassificationReport(include_roc_curve=True, include_pr_curve=True)

        # Perform cross-validation
        for train_ix, test_ix in cv_outer.split(X, y):
            print(".", end="")

            # Split data into training and testing sets
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]  
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            # Train the ensemble model
            training_time = time()
            estimador.fit(X_train, y_train)
            training_time = time() - training_time

            # Record results for this fold
            model_results.evaluate_and_store_results(
                best_model=estimador,
                best_params=None,  # No hyperparameters to store for the ensemble
                X_test=X_test,
                y_test=y_test,
                training_time=training_time
            )
        
        # Store results for this ensemble model
        new_results[name] = model_results.to_dict()
    
    return new_results
