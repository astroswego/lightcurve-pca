import numpy
from .utils import colvec

__all__ = [
    'Normalizer'
]

class Normalizer:
    def __init__(self, shifter='min', copy=True):
        if shifter in ['min', 'max']:
            self.shifter = shifter
        else:
            raise Exception("shifter must be either 'min' or 'max'")
        self.copy = copy

    def fit(self, X, y=None):
        X_min = numpy.amin(X, axis=1)
        X_max = numpy.amax(X, axis=1)
        self.range_ = colvec(X_max-X_min)
        self.shift_ = colvec(X_min if self.shifter == 'min' else X_max)

        return self

    def transform(self, X, y=None):
        X_new = numpy.array(X, copy=self.copy)
        X_new -= self.shift_
        X_new /= self.range_
        return X_new

    def inverse_transform(self, X):
        X_new = numpy.array(X, copy=self.copy)
        X_new *= self.range_
        X_new += self.shift_
        return X_new

    def get_params(self, deep=True):
        return {'shifter': self.shifter,
                'copy': self.copy}

    def set_params(self, **params):
        if 'shifter' in params:
            self.shifter = params['shifter']
        if 'copy' in params:
            self.copy = params['copy']
