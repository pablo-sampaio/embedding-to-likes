
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer

from imblearn import FunctionSampler


class IdentityTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


class TargetEncoderX(TargetEncoder):
    """
    Um "Target Encoder Misto" criado neste trabalho.
    
    Ele é aplicado a um "y" discreto (classificação), mas, internamente, faz um target enconding
    contínuo aplicado sobre uma coluna contínua (dada pelo parâmetro "y_continuous_full"). A ideia é 
    aplicar quando há uma coluna-alvo de classificação (y) que é derivada de uma coluna contínua 
    (y_continuous_full).

    Tecnicamente, a ideia é um target encoding contínuo cujo fit/fit_transform deveria receber 
    apenas a própria coluna contínua. Porém, esta implementação foi ajustada para permitir usar esta
    classe em um pipeline de um modelo que será treinado com a coluna de classificação, usando alguma
    forma de cross-validation.

    Para isso, esta implementação recebe o atributo contínuo de todo dataset no construtor, porém, 
    durante o fit/fit_tranform, ela usa (do "y_continuous_full") apenas as linhas correspondentes ao "y" 
    (de classificação) recebido. Para isso, é *fundamental manter a coerência entre os índices* de "y"
    e "y_continuous_full".
    
    Atenção: apesar de receber uma coluna com os valores de todo o dataset usado, não acontece vazamento
    de dados (que seria quando dados fora do conjunto de treinamento influenciam no treinamento)!
    - Apesar de receber TODA a coluna contínua de referência, o cálculo é efetivamente aplicado 
      usando apenas as linhas correspondentes ao "y" (de classificação)  recebido como parâmetro na 
      função "fit()" ou "fit_transform()", ou seja, usando apenas as instâncias de treinamento.
    - Após treinado (com fit/fit_transform), o transform() não depende mais do "y_continuous_full".
    
    """
    def __init__(self, y_continuous_full, **kwargs):
        super().__init__(target_type='continuous', **kwargs)
        self.y_continuous_full = y_continuous_full

    def fit(self, X, y):
        super().fit(X, self.y_continuous_full.loc[y.index])
        return self

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, y):
        return super().fit_transform(X, self.y_continuous_full.loc[y.index])
    

def one_hot_encoder_factory(columns, categories_per_column, drop_first=True):
   DROP_TYPE = 'first' if drop_first else 'if_binary'
   _transf_o=[
        ('cat', OneHotEncoder(categories=categories_per_column, drop=DROP_TYPE, sparse_output=False), columns)]
   ct = ColumnTransformer(transformers=_transf_o, remainder='passthrough')
   ct._simplified_name = f"OneHotEncoder(drop='{DROP_TYPE}')"
   return ct 

def target_encoder_binary_factory(columns):
    _transf_t1=[
        ('cat', TargetEncoder(shuffle=False, target_type='binary'), columns)]
    ct = ColumnTransformer(transformers=_transf_t1, remainder='passthrough')
    ct._simplified_name = f"TargetEncoder(target_type='binary')"
    return ct 

def target_encoder_continuous_factory(columns, y_continuous_full, use_log):
    #_dfy = dfy_full['Curtidas-Log'] if use_log else dfy_full['Curtidas']
    _dfy = np.log(y_continuous_full) if use_log else y_continuous_full
    _transf_t2=[
        ('cat', TargetEncoderX(_dfy, shuffle=False), columns)]
    ct = ColumnTransformer(transformers=_transf_t2, remainder='passthrough')
    ct._simplified_name = f"TargetEncoderX(target_type='continuous',use_log={use_log})"
    return ct 


# não pude usar lambda, porque não salva com np.load()
def _identity_sampling_function(X, y):
    return X, y

def identity_sampler_factory():
    fs = FunctionSampler(func=_identity_sampling_function)
    fs._simplified_name = f"IdentitySampler"
    return fs

