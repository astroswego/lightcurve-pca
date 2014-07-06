import numpy
from argparse import ArgumentParser, FileType
from sys import exit, stdin, stderr
from os import path, listdir
from sklearn.decomposition import PCA, ProbabilisticPCA
from lightcurve_pca.pca import make_pipeline, pca, reconstruct_lightcurve

def get_args():
    parser = ArgumentParser()
    input_group = parser.add_argument_group('Input')
    output_group = parser.add_argument_group('Output')
    pca_group = parser.add_argument_group('PCA')

    parser.add_argument('reconstruct_components', type=int, nargs='*',
        default=[1, 3, 7, 10],
        help='list of numbers of components to use in lightcurve reconstruction '
             '(default = 1 3 7 10)')
    input_group.add_argument('-l', '--lightcurves', type=str,
        help='File containing table of lightcurves, with star ID in first column')
    input_group.add_argument('--observations', type=str,
        help='Directory containing stellar observations. If nothing is provided, '
             'lightcurves are reconstructed without observed data '
             '(default = None)')
    input_group.add_argument('--periods', type=FileType('r'),
        default=None,
        help='File listing star periods. Only necessary if lightcurves are to be '
             'reconstructed from observed data '
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
    output_group.add_argument('--parameter-range', type=int, nargs=2,
        default=(1,2),
        help='Range (inclusive) of parameters to plot against periods '
             '(default = 1 2)')
    pca_group.add_argument('--n-components', type=float,
        default=None,
        help='Number of components to keep from the PCA. Keeps all components if '
             'none specified. If 0 < n_components < 1, selects the number of '
             'components such that the amount of variance that needs to be '
             'explained is greater than the percentage specified by n_components '
             '(default = None)')
    pca_group.add_argument('--method', type=str,
        choices=['PCA', 'ProbabilisticPCA'], default='PCA',
        help='Variant of PCA to use '
             '(default = PCA)')
    pca_group.add_argument('--whiten', action='store_true',
        default=False,
        help='Whiten the data '
             '(default = False)')

    args = parser.parse_args()

    method_choices = {'PCA': PCA,
                      'ProbabilisticPCA': ProbabilisticPCA}
    args.pipeline = make_pipeline(method_choices[args.method], **args.__dict__)

    return args

def main():
    args = get_args()

    with open(args.lightcurves, 'r') as f:
        first_line = f.readline().strip().split()
        cols = len(first_line)
        names = [first_line[0]] + [line.strip().split()[0] for line in f]
    
#    names = numpy.loadtxt(args.lightcurves,
#                          usecols=(0,), dtype=str, unpack=True)
    lightcurves = numpy.loadtxt(args.lightcurves,
                                usecols=range(1,cols), dtype=float)
    components, eigenvectors, col_std, col_mean = pca(lightcurves,
                                                      args.pipeline)
    if args.parameter_plots:
        for p in range(*args.parameter_range):
            name = 'PC{}'.format(p)
            parameter_plot(name=name, parameter=components[:,p],
                           periods=periods, output=args.parameter_plots)

    formatter = lambda x: args.fmt % x
    phases = numpy.arange(0, 1, 1/lightcurves.shape[1])
    for name, comps, lc in zip(names, components, lightcurves):
        print(name, ' '.join(map(formatter, comps)))
        if args.reconstruct_plots:
            reconstruct_lightcurve(name=name,
                lc=lc, phases=phases,
                components=comps, eigenvectors=eigenvectors,
                col_std=col_std, col_mean=col_mean,
                reconstruct_components=args.reconstruct_components,
                output=args.reconstruct_plots)

    if args.eigenvectors:
        numpy.savetxt(args.eigenvectors, eigenvectors, fmt=args.fmt)
