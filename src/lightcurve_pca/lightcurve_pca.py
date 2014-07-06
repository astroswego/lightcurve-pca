import numpy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, scale
from sklearn.decomposition import PCA
from argparse import ArgumentParser, FileType

__all__ = [
    'pca'
]

def get_args():
    parser = ArgumentParser()

    parser.add_argument(

def pca(X):
    pipeline = Pipeline([('Normalize', Normalizer()),
                         ('Standardize', StandardScaler()),
                         ('PCA', PCA())])
    mean_subtracted_X = scale(X, axis=1, with_mean=True, with_std=False)
    pcs = pipeline.fit(mean_subtracted_X).transform(mean_subtracted_X)
    eig = pipeline.named_steps['PCA'].components_

    return pcs, eig
