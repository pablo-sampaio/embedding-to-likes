
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn import metrics

from sklearn.base import clone


class ClassificationReport:
    def __init__(self, include_roc_curve=True, include_pr_curve=False):
        self.include_roc_curve = include_roc_curve
        self.include_pr_curve = include_pr_curve

        self.best_params = []
        self.best_models = []
        self.training_times = []

        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_score_list = []
        self.auc_roc_score_list = []
        self.auc_pr_score_list = []

        self.roc_curves = []  # pairs (list of FPRs, list of TPRs) per fold
        self.pr_curves  = []  # pairs (list of recalls, list of precisions) per fold
        
        self.num_fold = 0

    def _check_consistency(self):
        assert len(self.best_params) == self.num_fold + 1
        assert len(self.best_models) == self.num_fold + 1
        assert len(self.training_times) == self.num_fold + 1
        assert len(self.accuracy_list) == self.num_fold + 1
        assert len(self.precision_list) == self.num_fold + 1
        assert len(self.recall_list) == self.num_fold + 1
        assert len(self.f1_score_list) == self.num_fold + 1
        assert len(self.auc_roc_score_list) == self.num_fold + 1
        assert len(self.auc_pr_score_list) == self.num_fold + 1
        assert len(self.roc_curves) == self.num_fold + 1
        assert len(self.pr_curves) == self.num_fold + 1

    def evaluate_and_store_results(self, best_model, best_params, X_test, y_test, training_time):
        '''
        Receive a trained binary classification model, then calculates and stores metrics.
        '''
        self.best_params.append(best_params)
        self.best_models.append(best_model)
        self.training_times.append(training_time)
        
        y_pred = best_model.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)

        # Armazenando métricas deste fold
        self.accuracy_list.append(accuracy)
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_score_list.append(f1)

        model_predicts_probability = hasattr(best_model, "predict_proba")
        
        if model_predicts_probability:
            # Probabilidades da classe positiva
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]  
            
            # Calcula auc-ROC (area under the curve, for ROC curve)
            aucroc_score = metrics.roc_auc_score(y_test, y_pred_proba)
            self.auc_roc_score_list.append(aucroc_score)

            # Calcula auc-PR (area under the curve, for PR curve)
            # https://datascience.stackexchange.com/questions/9003/when-do-i-have-to-use-aucpr-instead-of-auroc-and-vice-versa
            aucpr_score = metrics.average_precision_score(y_test, y_pred_proba) 
            self.auc_pr_score_list.append(aucpr_score)
        else:
            print(f"Warning: Modelo não permite calcular curvas ROC ou PR.")
            self.auc_roc_score_list.append(0)
            self.auc_pr_score_list.append(0)

        if self.include_roc_curve and model_predicts_probability:
            # Valores usados para plotar a curva ROC
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)  # Calcula FPR (false positive rate) e TPR (true positive rate)
            self.roc_curves.append((fpr, tpr))
        else:
            self.roc_curves.append(None)

        if self.include_pr_curve and model_predicts_probability:
            # Valores usados para plotar para a curva PR
            precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, y_pred_proba)  # Calcula Precision e Recall
            self.pr_curves.append((recalls, precisions))
        else:          
            self.pr_curves.append(None)
        
        self._check_consistency()
        self.num_fold += 1

    def to_dict(self):
        result = {
            "best_parameters": self.best_params,
            "best_models": self.best_models,
            "average_training_time": np.mean(self.training_times),

            "accuracy_mean": np.mean(self.accuracy_list),
            "accuracy_std": np.std(self.accuracy_list),
            "precision_mean": np.mean(self.precision_list),
            "precision_std": np.std(self.precision_list),
            "recall_mean": np.mean(self.recall_list),
            "recall_std": np.std(self.recall_list),
            "f1_score_mean": np.mean(self.f1_score_list),
            "f1_score_std": np.std(self.f1_score_list),
            "f1_score_list": self.f1_score_list,
            "auc_roc_mean": np.mean(self.auc_roc_score_list),
            "auc_roc_std": np.std(self.auc_roc_score_list),
            "auc_pr_mean": np.mean(self.auc_pr_score_list),
            "auc_pr_std": np.std(self.auc_pr_score_list),

            "roc_curve_list": self.roc_curves,
            "pr_curve_list": self.pr_curves,            
        }
        return result


def nested_cross_validation_grid_search(lista_modelos, X, y, cv_outer, cv_inner):
    """
    Esta função otimiza e treina vários modelos e avalia seu desempenho usando métricas como acurácia, 
    precisão, revocação e F1-score.

    Ela recebe vários tipos de modelos, cada um com um dicionário informando os hiperparâmetros e a lista de 
    valores, para fazer uma otimização baseada em grid search.

    Para cada modelo, aplica uma nested cross-fold validation (nested CV):
    - Um k-fold CV externo é usado para dividir em dados de treinamento e teste.
    - Nos dados de treinamento é aplicado um grid search que avalia com outro k-fold CV. (Este CV interno
      automaticamente divide os dados de treinamento em treinamento e validação,).
    
    A função retorna uma lista com as melhores configurações de cada modelo (uma para cada fold externo) e
    também os valores de várias métricas de classificação do modelo e os dados das curvas ROC e PR.

    Args:
        lista_modelos: Uma lista de dicionários contendo informações sobre os modelos a serem treinados.
        X: Conjunto de dados.
        y: Rótulos.

    Returns:
        dict: Um dicionário com os dados das melhores configurações de cada modelo (uma para cada divisão do
        CV externo) e as métricas de desempenho de cada uma dessas configurações (após o modelo ser retreinado
        e testado com a divisão dos dados feita pelo CV externo).
    """

    print(f"\n\n\n **** RESULTADO DOS MODELOS ****\n")

    resultados_gerais = {}  # Dicionário para armazenar os resultados desta iteração

    # Configurando a busca em grade dentro de cada iteração da validação cruzada externa
    for mdl in lista_modelos:
        nome_do_modelo = mdl["nome_do_modelo"]
        estimador_base = mdl.get('estimador')
        parametros = mdl.get('parametros')

        print(f"Treinando modelo {nome_do_modelo} ", end="")

        # Instancia a classe ModelResultsCV para armazenar os resultados
        model_results = ClassificationReport(True, True)

        # Executando a validação cruzada
        for train_ix, test_ix in cv_outer.split(X, y):
            print(".", end="")

            # Separa em dados de treinamento-validação 
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]  
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            # Roda grid search, capturando tempo de treinamento
            grid_search = GridSearchCV(estimador_base, 
                                       parametros, 
                                       scoring='f1', 
                                       cv=cv_inner,
                                       n_jobs=1)
            
            # Inicio do treinamento
            tempo_treinamento = time.time()
            
            # Treinamento 1 (múltiplos) - grid search
            modelo_treinado = grid_search.fit(X_train, y_train)

            modelo_treinado = clone(grid_search.best_estimator_)
            modelo_treinado.set_params(**grid_search.best_params_)  # só por garantia...

            if nome_do_modelo == 'Support Vector Machine':
                modelo_treinado.set_params(predictor__probability=True)  # não é setado antes, porque deixa o treinamento lento
            
            # Treinamento 2 - retreina com todos os dados de treinamento-validação            
            modelo_treinado = modelo_treinado.fit(X_train, y_train)

            # Fim do treinamento
            tempo_treinamento = time.time() - tempo_treinamento

            # Avaliação do modelo (com os melhores parâmetros encontrados) no conjunto de teste (separado no fold externo)
            model_results.evaluate_and_store_results(
                best_model=modelo_treinado,
                best_params=grid_search.best_params_,
                X_test=X_test,
                y_test=y_test,
                training_time=tempo_treinamento
            )
        
        print("\n-- coletando e armazenando resultados --\n")

        # Armazena os resultados no dicionário geral
        resultados_gerais[nome_do_modelo] = model_results.to_dict()

        # Exibe as métricas principais
        print(f" - Acurácia   : {resultados_gerais[nome_do_modelo]['accuracy_mean']:.4f} +/- {resultados_gerais[nome_do_modelo]['accuracy_std']:.4f}")
        print(f" - Precisão   : {resultados_gerais[nome_do_modelo]['precision_mean']:.4f} +/- {resultados_gerais[nome_do_modelo]['precision_std']:.4f}")
        print(f" - Revocação  : {resultados_gerais[nome_do_modelo]['recall_mean']:.4f} +/- {resultados_gerais[nome_do_modelo]['recall_std']:.4f}")
        print(f" - F1 - Score : {resultados_gerais[nome_do_modelo]['f1_score_mean']:.4f} +/- {resultados_gerais[nome_do_modelo]['f1_score_std']:.4f}")
        print(f" - ROC - AUC  : {resultados_gerais[nome_do_modelo]['auc_roc_mean']:.4f} +/- {resultados_gerais[nome_do_modelo]['auc_roc_std']:.4f}")
        print(f" - PR - AUC   : {resultados_gerais[nome_do_modelo]['auc_pr_mean']:.4f} +/- {resultados_gerais[nome_do_modelo]['auc_pr_std']:.4f}")
        print(f" - Tempo médio de treinamento: {resultados_gerais[nome_do_modelo]['average_training_time']:.2f} segundos\n")
        print('=' * 50, '\n')

    print("Terminado em", time.strftime('%d/%m/%Y %H:%M:%S', time.localtime()))

    return resultados_gerais


def plot_roc_curve(resultados_gerais, model_name, fold, figsize=(10, 8), show_plot=True, save_path=None, plot_ref_line=True):
    """
    Plota as curvas ROC para os modelos treinados.

    Args:
        resultados_gerais: Dicionário com os resultados dos modelos.
        model_name: Nome do modelo a ser plotado.
        fold: Número do fold para o qual a curva ROC será plotada.
        figsize: Tamanho da figura (largura, altura). Default é (10, 8).
        show_plot: Se True, exibe o gráfico automaticamente. Default é True.
        save_path: Caminho para salvar o gráfico como imagem. Se None, não salva. Default é None.

    Returns:
        plt: Objeto matplotlib.pyplot para personalizações adicionais.
    """
    if show_plot or save_path:
        # Configura o tamanho do gráfico
        plt.figure(figsize=figsize)

    # Itera sobre os modelos e plota a curva ROC do modelo especificado
    for mdl in resultados_gerais.keys():
        if mdl == model_name:
            roc_curves = resultados_gerais[mdl]['roc_curve_list']
            if fold < len(roc_curves) and roc_curves[fold] is not None:
                fpr, tpr = roc_curves[fold]
                plt.plot(fpr, tpr, label=f'{mdl} (Fold {fold + 1})')
            else:
                print(f"Warning: Fold {fold} não possui curva ROC para o modelo {mdl}.")
    
    if plot_ref_line:
        # Adiciona a linha de referência (diagonal)
        plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo (FPR)')
    plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
    plt.title(f'Curva ROC - {model_name} (Fold {fold + 1})')
    plt.legend(loc='lower right')

    # Salva o gráfico, se necessário
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")

    # Exibe o gráfico, se necessário
    if show_plot:
        plt.show()


def plot_pr_curve(resultados_gerais, model_name, fold, figsize=(10, 8), show_plot=True, save_path=None):
    """
    Plota as curvas PR para os modelos treinados.

    Args:
        resultados_gerais: Dicionário com os resultados dos modelos.
        model_name: Nome do modelo a ser plotado.
        fold: Número do fold para o qual a curva PR será plotada.
        show_plot: Se True, exibe o gráfico automaticamente. Default é True.
        save_path: Caminho para salvar o gráfico como imagem. Se None, não salva. Default é None.

    Returns:
        plt: Objeto matplotlib.pyplot para personalizações adicionais.
    """
    if show_plot or save_path:
        # Configura o tamanho do gráfico
        plt.figure(figsize=figsize)

    # Itera sobre os modelos e plota a curva PR do modelo especificado
    for mdl in resultados_gerais.keys():
        if mdl == model_name:
            pr_curves = resultados_gerais[mdl]['pr_curve_list']
            if fold < len(pr_curves) and pr_curves[fold] is not None:
                recalls, precisions = pr_curves[fold]
                plt.plot(recalls, precisions, label=f'{mdl} (Fold {fold + 1})')
            else:
                print(f"Warning: Fold {fold} não possui curva PR para o modelo {mdl}.")
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title(f'Curva PR - {model_name} (Fold {fold + 1})')
    plt.legend(loc='lower right')

    # Salva o gráfico, se necessário
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")

    # Exibe o gráfico, se necessário
    if show_plot:
        plt.show()


class RandomClassifier(BaseEstimator):
    def __init__(self, true_class_prob=0.5):
        self.class1_prob = true_class_prob

    def fit(self, X, y):
        return self

    def predict(self, X):
        y_pred = np.random.choice([0, 1], size=len(X), p=[1.0-self.class1_prob, self.class1_prob])
        return y_pred
    
    def predict_proba(self, X):
        y_pred_proba = np.zeros((len(X), 2))
        y_pred_proba[:, 0] = 1.0 - self.class1_prob
        y_pred_proba[:, 1] = self.class1_prob
        return y_pred_proba


def random_classifier_results(y_true, classifier_true_prob=0.5, trials=100):
    """
    Generates random predictions and evaluates them against the true labels.
    
    Parameters:
    - y_true: True labels
    - classifier_true_prob: Probability of predicting class 1 (true)
    - trials: Number of trials to run
    
    Returns:
    - None (prints results)
    """
    classifier = RandomClassifier(classifier_true_prob)
    report = ClassificationReport(False, False)
    
    for i in range(trials):
        report.evaluate_and_store_results(classifier, None, y_true, y_true, 0.0)
    
    return report.to_dict()


from sklearn.base import ClassifierMixin

# Não está sendo usado, no momento
class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, find_best_threshold=False):
        self.base_model = base_model
        self.threshold = 0.5 # Default threshold for binary classification
        self.find_best_threshold = find_best_threshold
    
    def fit(self, X, y):
        self.base_model.fit(X, y)
        if self.find_best_threshold:
            self.find_best_threshold(X, y)
        return self
    
    def predict(self, X):
        y_probs = self.base_model.predict_proba(X)[:, 1]
        return (y_probs > self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
    
    def set_threshold(self, new_threshold):
        """Update the decision threshold."""
        self.threshold = new_threshold
    
    def find_best_threshold(self, X_val, y_val, thresholds=np.linspace(0.1, 0.9, 20)):
        """Finds the threshold that maximizes the F1-score."""
        y_probs = self.base_model.predict_proba(X_val)[:, 1]
        best_threshold = self.threshold
        best_f1 = 0
        
        for t in thresholds:
            y_pred = (y_probs > t).astype(int)
            f1 = metrics.f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        self.threshold = best_threshold
        return best_threshold, best_f1
