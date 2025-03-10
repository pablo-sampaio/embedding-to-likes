import numpy as np

from sklearn import metrics
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold


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
        pred_proba = np.array([estimator.predict_proba(X) for estimator in self.estimators])
        pred_proba = np.mean(pred_proba, axis=0) / len(self.estimators)
        return pred_proba
    

def select_best_models_per_fold(results):
    '''
    Seleciona a melhor modelo (de qualquer tipo) com seus parâmetros, para cada fold externo
    (Lembrando que cada modelo tem uma "configuração ótima" por fold externo).
    '''
    best_models_per_fold = []
    for fold in range(5):
        best_f1 = 0.0
        best_model = None
        best_model_name = None
        for model_name in results.keys():
            if results[model_name]['F1_score_list'][fold] > best_f1:
                best_f1 = results[model_name]['F1_score_list'][fold]
                best_model = clone( results[model_name]['melhores_modelos'][fold] )
                best_model.set_params(**results[model_name]['melhores_parametros'][fold])
                best_model_name = model_name
        best_models_per_fold.append((f"{best_model_name} #{fold+1}", best_model))
    return best_models_per_fold


def select_best_models_of_type_fn(model_name):
    def _select_models(results):
        return [ (f"{model_name} #{fold}", clone(model)) for fold, model in enumerate(results[model_name]['melhores_modelos']) ]
    return _select_models


def train_ensemble(select_models_fn, results, X, y, name_prefix="ensemble"):
    model_list = select_models_fn(results)
    
    _estimador_hard = MyVotingClassifier(estimators=model_list, voting='hard')
    _estimador_soft = MyVotingClassifier(estimators=model_list, voting='soft') 

    new_results = { } #deepcopy(results)
    
    # Executando a validação cruzada
    for name, estimador in [(name_prefix+'-hard-vote', _estimador_hard), (name_prefix+'-soft-vote', _estimador_soft)]:
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []        

        # A mesma CV usada no treinamento de cada modelo
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_ix, test_ix in cv_outer.split(X, y):
            print(".", end="")

            # Separa em dados de treinamento-validação 
            # (obs.: eles serão novamente divididos internamente pelo cv do grid-search)
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]  
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            estimador.fit(X_train, y_train)

            y_pred = estimador.predict(X_test)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            precisions = metrics.precision_score(y_test, y_pred)
            recalls = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)

            # Armazenando métricas deste fold
            accuracy_list.append(accuracy)
            precision_list.append(precisions)
            recall_list.append(recalls)
            f1_score_list.append(f1)
        
        new_results[name] = {
            "Acurácia_mean": np.mean(accuracy_list),
            "Acurácia_std": np.std(accuracy_list),
            "Precisão_mean": np.mean(precision_list),
            "Precisão_std": np.std(precision_list),
            "Revocação_mean": np.mean(recall_list),
            "Revocação_std": np.std(recall_list),
            "F1_score_mean": np.mean(f1_score_list),
            "F1_score_std": np.std(f1_score_list),
            "F1_score_list": f1_score_list,
            "melhores_modelos": " + ".join([name for name, model in model_list])  # só a lista de nomes
        }
    
    return new_results
