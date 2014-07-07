import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from .preprocessing import Normalizer
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

def pca(X, pipeline, reconstruct_orders):
#    mean_subtracted_X = scale(X, axis=1, with_mean=True, with_std=False)
#    pcs = pipeline.fit(mean_subtracted_X).transform(mean_subtracted_X)
    components = pipeline.fit_transform(X)
    eig = pipeline.named_steps['PCA'].components_

    reconstructed_lightcurves = numpy.empty((X.shape[0],
                                             len(reconstruct_orders),
                                             X.shape[1]),
                                            dtype=float)
    for i, n in enumerate(reconstruct_orders):
        pipeline.set_params(PCA__n_components=n)
        comp = pipeline.fit_transform(X)
        reconstructed_lightcurves[:, i, :] = pipeline.inverse_transform(comp)

    return components, eig, reconstructed_lightcurves

def plot_lightcurve(name, lc, phases, reconstructs, reconstruct_orders,
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
            color='black', label='Light Curve')

    # Plot observed points
    ## TODO

    # Plot reconstructed lightcurves
    for n, rec in zip(reconstruct_orders, reconstructs):
        ax.plot(two_phases, numpy.hstack((rec, rec)),
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
