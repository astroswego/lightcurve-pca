import numpy
from argparse import ArgumentParser, FileType
from sys import exit, stdin, stderr
from os import path, listdir
from sklearn.decomposition import PCA, ProbabilisticPCA
from lightcurve_pca.pca import (make_pipeline, pca, plot_lightcurve,
                                parameter_plot)
from lightcurve_pca.utils import pmap

def get_args():
    parser = ArgumentParser()
    input_group = parser.add_argument_group('Input')
    output_group = parser.add_argument_group('Output')
    pca_group = parser.add_argument_group('PCA')
    plot_group = parser.add_argument_group('Plot')

    parser.add_argument('reconstruct_orders', type=int, nargs='*',
        default=[1, 3, 7, 10],
        help='list of numbers of components to use in lightcurve '
             'reconstruction '
             '(default = 1 3 7 10)')
    parser.add_argument('-p', '--processes', type=int,
        default=1, metavar='N',
        help='number of stars to plot in parallel '
             '(default = 1)')
    input_group.add_argument('-l', '--lightcurves', type=str,
        help='File containing table of lightcurves, with star ID in first '
             'column')
    input_group.add_argument('--observations', type=str,
        help='Directory containing stellar observations. If nothing is '
             'provided, lightcurves are reconstructed without observed data '
             '(default = None)')
    input_group.add_argument('--observation-extension', type=str,
        default='.dat',
        help='Extension of observation files '
             '(default = .dat)')
    input_group.add_argument('--usecols', type=int, nargs=3,
        default=range(3), metavar=['TIME', 'MAG', 'ERR'],
        help='Columns to use in observation files '
             '(default = 0 1 2)')
    input_group.add_argument('--periods', type=FileType('r'),
        default=None,
        help='File listing star periods. Only necessary if lightcurves are to '
             'be reconstructed from observed data '
             '(default = None)')
    output_group.add_argument('-f', '--fmt', type=str,
        default='%.5f',
        help='format specifier for output tables')
    output_group.add_argument('--eigenvalues', type=str,
        help='File to output eigenvalues in '
             '(default = None)')
    output_group.add_argument('--eigenvectors', type=str,
        help='File to output eigenvectors in '
             '(default = None)')
    output_group.add_argument('--reconstruct-plots', type=str,
        help='Directory to output reconstructed lightcurves '
             '(default = None)')
    output_group.add_argument('--parameter-plots',  type=str,
        help='Directory to output parameter plots '
             '(default = None)')
    pca_group.add_argument('--n-components', type=float,
        default=None,
        help='Number of components to keep from the PCA. Keeps all components '
             'if none specified. If 0 < n_components < 1, selects the number '
             'of components such that the amount of variance that needs to be '
             'explained is greater than the percentage specified by '
             'n_components '
             '(default = None)')
    pca_group.add_argument('--method', type=str,
        choices=['PCA', 'ProbabilisticPCA'], default='PCA',
        help='Variant of PCA to use '
             '(default = PCA)')
    pca_group.add_argument('--whiten', action='store_true',
        default=False,
        help='Whiten the data '
             '(default = False)')
    plot_group.add_argument('--parameter-range', type=int, nargs=2,
        default=(1,2),
        help='Range (inclusive) of parameters to plot against periods '
             '(default = 1 2)')
    plot_group.add_argument('--plot-original-lightcurve',
        default=False,
        help='Include the original lightcurve in the plot '
             '(default = False)')

    args = parser.parse_args()

    method_choices = {'PCA': PCA,
                      'ProbabilisticPCA': ProbabilisticPCA}
    args.pipeline = make_pipeline(method_choices[args.method], **vars(args))
    if args.periods is not None:
        periods = {name: float(period) for (name, period)
                   in (line.strip().split() for line
                   # generalize to all whitespace instead of just spaces
                   in args.periods if ' ' in line)}
        args.periods.close()
        args.periods = periods

    return args

def main():
    args = get_args()

    with open(args.lightcurves, 'r') as f:
        first_line = f.readline().strip().split()
        cols = len(first_line)
        names = [first_line[0]] + [line.strip().split()[0] for line in f]

    lightcurves = numpy.loadtxt(args.lightcurves,
                                usecols=range(1,cols), dtype=float)
    components, eigenvectors, reconstructs =  pca(lightcurves, args.pipeline,
                                                  args.reconstruct_orders)
    periods = [args.periods[name] for name in names]
    if args.eigenvectors:
        numpy.savetxt(args.eigenvectors, eigenvectors, fmt=args.fmt)

    if args.parameter_plots:
        for p in range(args.parameter_range[0], args.parameter_range[1]+1):
            name = 'PC{}'.format(p)
            parameter_plot(name=name, parameter=components[:,p],
                           periods=periods, output=args.parameter_plots)

    formatter = lambda x: args.fmt % x
    phases = numpy.arange(0, 1, 1/lightcurves.shape[1])
    pmap(process_star,
         zip(names, periods, components, lightcurves, reconstructs),
         reconstruct_orders=args.reconstruct_orders,
         processes=args.processes,
         callback=_star_printer(args.fmt),
         phases=phases,
         plot_original_lightcurve=args.plot_original_lightcurve,
         output=args.reconstruct_plots,
         observations=args.observations,
         observation_extension=args.observation_extension)

def process_star(star, **kwargs):
    name, period, components, lightcurve, reconstruct = star
    plot_lightcurve(name=name, lc=lightcurve, reconstructs=reconstruct,
                    period=period,
                    **kwargs)
    return name, components

def _star_printer(fmt):
    return lambda name_and_components: _print_star(name_and_components, fmt)

def _print_star(name_and_components, fmt):
    name, components = name_and_components
    print(name, ' '.join(fmt % comp for comp in components))
