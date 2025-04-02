
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn import metrics

from sklearn.base import clone


class ModelResultsCV:
    def __init__(self, cv_object, include_roc_curve=True, include_pr_curve=False):
        self.cv = cv_object

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
        assert len(self.accuracy_list) == self.num_fold+1
        # TODO: fazer o mesmo com as outras listas...

    def record_fold_best_model_evaluation(self, best_model, best_params, X_test, y_test, training_time):
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
        auc_mean = np.mean(self.auc_score_list)
        auc_std = np.std(self.auc_score_list)
        aucpr_mean = np.mean(self.aucpr_score_list)
        aucpr_std = np.std(self.aucpr_score_list)

        result = {
            "cv_model": self.cv_object,

            "accuracy_mean": np.mean(self.accuracy_list),
            "accuracy_std": np.std(self.accuracy_list),
            "precision_mean": np.mean(self.precision_list),
            "precision_std": np.std(self.precision_list),
            "recall_mean": np.mean(self.recall_list),
            "recall_std": np.std(self.recall_list),
            "f1_score_mean": np.mean(self.f1_score_list),
            "f1_score_std": np.std(self.f1_score_list),
            "f1_score_list": self.f1_score_list,
            "aucROC_mean": auc_mean,
            "aucROC_std": auc_std,
            "aucPR_mean": aucpr_mean,
            "aucPR_std": aucpr_std,

            "roc_curve_list": self.roc_curves,
            "pr_curve_list": self.pr_curves,
            
            "average_training_time": np.mean(self.training_time),
            "best_parameters": self.best_model_params,
        }
        return result


def nested_cross_validation_grid_search(lista_modelos, X, y, k_folds_outer=5, k_folds_inner=5, repetitions=1, rand_state=42):
    """
    Esta função treina vários modelos e avalia seu desempenho usando métricas como acurácia, 
    precisão, revocação e F1-score.

    Ela recebe vários tipos de modelos, cada um com um dicionário informando os hiperparâmetros e a lista de 
    valores, para fazer uma otimização baseada em grid search.

    Para cada modelo, aplica uma nested cross-fold validation:
    - Um k-fold CV externo é usado para dividir em dados de treinamento e teste.
    - Nos dados de treinamento é aplicado um k-fold interno, no qual é aplicado o grid search.
    - A saída (por modelo) é uma lista com as melhores configurações do modelo para cada fold externo, 
      junto com os valores das métricas (retreinadas e testadas com a divisão feita pelo CV externo).

    Também plota as curvas ROC e PR para cada modelo.

    Args:
        lista_modelos: Uma lista de dicionários contendo informações sobre os modelos a serem treinados.
        X: Conjunto de dados.
        y: Rótulos.

    Returns:
        dict: Um dicionário com as métricas de desempenho de cada modelo e dados dos melhores modelos.
    """

    print(f"\n\n\n **** RESULTADO DOS MODELOS + CURVAS ROC E PR ****\n")

    # Lista para armazenar os valores de fpr e tpr de cada modelo (para a curva ROC)
    roc_fpr_list = []
    roc_tpr_list = []

    # Lista para armazenar os valores de precision e recall de cada modelo (para a curva PR)
    pr_precision_list = []
    pr_recall_list = []
        
    resultados_gerais = {}  # Dicionário para armazenar os resultados desta iteração

    # Configurando a busca em grade dentro de cada iteração da validação cruzada externa
    for mdl in lista_modelos:
        nome_do_modelo = mdl["nome_do_modelo"]
        estimador_base = mdl.get('estimador')
        parametros = mdl.get('parametros')

        # Listas para armazenar métricas de interesse em cada fold
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        auc_score_list = []
        aucpr_score_list = []

        print(f"Treinando modelo {nome_do_modelo} ", end="")

        # Configurando a validação cruzada externa
        #cv_outer = StratifiedKFold(n_splits=k_folds_outer, shuffle=True, random_state=rand_state)
        cv_outer = RepeatedStratifiedKFold(n_repeats=repetitions, n_splits=k_folds_outer, random_state=rand_state)

        # Executando a validação cruzada
        tempos_de_treinamento = []
        best_model_params = []
        best_trained_models = []

        for train_ix, test_ix in cv_outer.split(X, y):
            print(".", end="")

            # Separa em dados de treinamento-validação 
            # (obs.: eles serão novamente divididos internamente pelo cv do grid-search)
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]  
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            # Roda grid search, capturando tempo de treinamento
            grid_search = GridSearchCV(estimador_base, parametros, 
                                       scoring='f1', 
                                       cv=StratifiedKFold(n_splits=k_folds_inner, shuffle=True),
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

            tempos_de_treinamento.append(tempo_treinamento)

            best_model_params.append(grid_search.best_params_)            
            best_trained_models.append(modelo_treinado)

            # Avaliação do modelo (com os melhores parâmetros encontrados) no conjunto de teste (separado no fold externo)
            y_pred = modelo_treinado.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)

            # Armazenando métricas deste fold
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1)
            
            if hasattr(modelo_treinado, "predict_proba"):
                # Probabilidades da classe positiva
                y_pred_proba = modelo_treinado.predict_proba(X_test)[:, 1]  
                
                # Calcula auc-ROC (area under the curve, for ROC curve)
                auc_score = metrics.roc_auc_score(y_test, y_pred_proba)
                auc_score_list.append(auc_score)

                # Valores usados para plotar a curva ROC
                fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)  # Calcula FPR (false positive rate) e TPR (true positive rate)
                roc_fpr_list.append(fpr)
                roc_tpr_list.append(tpr)

                # Calcula auc-PR (area under the curve, for PR curve)
                # https://datascience.stackexchange.com/questions/9003/when-do-i-have-to-use-aucpr-instead-of-auroc-and-vice-versa
                aucpr_score = metrics.average_precision_score(y_test, y_pred_proba) 
                aucpr_score_list.append(aucpr_score)

                # Valores usados para plotar para a curva PR
                precisions, recalls, thresholds = metrics.precision_recall_curve(y_test, y_pred_proba)  # Calcula Precision e Recall
                pr_precision_list.append(precisions)
                pr_recall_list.append(recalls)
            else:
                #print('x', end='')
                print(f"Modelo {nome_do_modelo} não permite calcular curva.")
                auc_score_list.append(0)
                aucpr_score_list.append(0)
                pr_precision_list.append([])
                pr_recall_list.append([])
        
        print("\n-- coletando e armazenando resultados --\n")

        # Calculando as médias e desvios padrão das métricas
        accuracy_mean = np.mean(accuracy_list)
        accuracy_std = np.std(accuracy_list)
        precision_mean = np.mean(precision_list)
        precision_std = np.std(precision_list)
        recall_mean = np.mean(recall_list)
        recall_std = np.std(recall_list)
        f1_score_mean = np.mean(f1_score_list)
        f1_score_std = np.std(f1_score_list)

        auc_mean = np.mean(auc_score_list)
        auc_std = np.std(auc_score_list)
        aucpr_mean = np.mean(aucpr_score_list)
        aucpr_std = np.std(aucpr_score_list)

        #print(f" - Modelo     : {nome_do_modelo}")
        print(f" - Acurácia   : {accuracy_mean:.4f} +/- {accuracy_std:.5f}")
        print(f" - Precisão   : {precision_mean:.4f} +/- {precision_std:.5f}")
        print(f" - Revocação  : {recall_mean:.4f} +/- {recall_std:.5f}")
        print(f" - F1 - Score : {f1_score_mean:.4f} +/- {f1_score_std:.5f}")
        print(f" - ROC - AUC  : {auc_mean:.4f} +/- {auc_std:.5f}")
        print(f" - PR - AUC   : {aucpr_mean:.4f} +/- {aucpr_std:.5f}")
        print(f" - Tempo médio de treinamento: {np.mean(tempos_de_treinamento):.2f} segundos\n")
        print('=' * 50, '\n')

        #resultados_iteracao[mdl.get('nome_do_modelo')]  = {
        resultados_gerais[nome_do_modelo]  = {
            "Acurácia_mean": accuracy_mean,
            "Acurácia_std": accuracy_std,
            "Precisão_mean": precision_mean,
            "Precisão_std": precision_std,
            "Revocação_mean": recall_mean,
            "Revocação_std": recall_std,
            "F1_score_mean": f1_score_mean,
            "F1_score_std": f1_score_std,
            "aucROC_mean": auc_mean,
            "aucROC_std": auc_std,
            "aucPR_mean": aucpr_mean,
            "aucPR_std": aucpr_std,
            "tempo_medio_treinamento": np.mean(tempos_de_treinamento),
            "F1_score_list": f1_score_list,
            "melhores_parametros": best_model_params, 
            "melhores_modelos": best_trained_models 
        }

    print("Terminado em", time.strftime('%d/%m/%Y %H:%M:%S', time.localtime()))

    # Tentar passar para uma função à parte, com os dados salvos nos resultados
    # Depois de treinar todos os modelos:

    # Plota a curva ROC geral
    plt.figure(figsize=(10, 8)) 
    for fpr, tpr, mdl in zip(roc_fpr_list, roc_tpr_list, lista_modelos):
        plt.plot(fpr, tpr, label='%s ROC' % mdl["nome_do_modelo"])
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('CURVA ROC')
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0))  # Posiciona a legenda
    plt.show()

    # Plota a curva PR geral - não sei se está correta
    # ver https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    # Ideia: plotar a média das curvas PR
    #'''
    plt.figure(figsize=(10, 8))
    for precisions, recalls, mdl in zip(pr_precision_list, pr_recall_list, lista_modelos):
        plt.plot(recalls, precisions, label='%s PR' % mdl["nome_do_modelo"])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('CURVA PR')
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0))  # Posiciona a legenda
    plt.show()
    #'''

    return resultados_gerais



