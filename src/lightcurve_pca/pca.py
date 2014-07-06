import numpy
import matplotlib.pyplot as plt
from os import path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, scale
from sklearn.decomposition import PCA
from .utils import make_sure_path_exists

__all__ = [
    'make_pipeline',
    'pca',
    'reconstruct_lightcurve'
]

def make_pipeline(PCA_method, n_components, whiten, **kwargs):
    pipeline = Pipeline([('Normalize', Normalizer()),
                         ('Standardize', StandardScaler()),
                         ('PCA', PCA_method(n_components=n_components,
                                            whiten=whiten))])

    return pipeline

def pca(X, pipeline):
#    mean_subtracted_X = scale(X, axis=1, with_mean=True, with_std=False)
#    pcs = pipeline.fit(mean_subtracted_X).transform(mean_subtracted_X)
    components = pipeline.fit_transform(X)
    eig = pipeline.named_steps['PCA'].components_
    std = pipeline.named_steps['Standardize'].std_
    mean = pipeline.named_steps['Standardize'].mean_
    
    return components, eig, std, mean

def reconstruct_lightcurve(name, lc, phases, components, eigenvectors,
                           col_std=0.0, col_mean=0.0,
                           reconstruct_components=(7,),
                           period=0.0, data=None, legend=True,
                           output='.', filetype='.png'):
    fig = plt.figure()
    ax = fig.gca()
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_xlim(0, 2)
    two_phases = numpy.hstack((phases, 1+phases))

    # Plot the original light curve
    ax.plot(two_phases, numpy.hstack((lc, lc)),
            color='green', label='Light Curve')

    # Plot observed points
    ## TODO

    # Plot reconstructed lightcurves
    for n in reconstruct_components:
        rec_lc = numpy.dot(components[:n], eigenvectors[:n,:])
        ax.plot(two_phases, numpy.hstack((rec_lc, rec_lc)),
                linewidth=1.0, label='{} components'.format(n))

    if legend:
        ax.legend(loc='best')

    ax.set_xlabel('Phase ({0:0.7} day period)'.format(period))
    ax.set_ylabel('Magnitude')

    ax.set_title(name)
    fig.tight_layout(pad=0.1)
    make_sure_path_exists(output)
    fig.savefig(path.join(output, name+filetype))
    fig.clf()

def parameter_plot(name, parameter, periods,
                   logscale=True,
                   output='.', filetype='.png'):
    fig = plt.figure()
    ax = fig.gca()
    ax.grid(True)

    # Plot the parameter
    P = numpy.log10(periods) if logscale else periods
    ax.scatter(P, parameter,
               color='black')

    xlabel = 'log(P)' if logscale else 'Period (days)'
    ylabel = name
    title  = name + ' vs ' + ('log(P)' if logscale else 'Period')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout(pad=0.1)
    make_sure_path_exists(output)
    fig.savefig(path.join(output, title.replace(' ', '_') + filetype))
    fig.clf()
