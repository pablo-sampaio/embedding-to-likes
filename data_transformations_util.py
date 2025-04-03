
from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

