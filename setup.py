#!/usr/bin/env python3
"""lightcurve-pca: variable star light curve principal component analysis tool

lightcurve-pca is a tool for performing principal component analysis (PCA) on
the lightcurves of variable stars.
"""

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Programming Language :: Python
Programming Language :: Python :: 3
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: OS Independent
Topic :: Scientific/Engineering :: Astronomy
"""

MAJOR    = 0
MINOR    = 1
MICRO    = 0
ISRELEASED = False
PRERELEASE = 1
VERSION    = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

def get_version_info():
    FULLVERSION = VERSION

    if not ISRELEASED:
        FULLVERSION += '-pre' + str(PRERELEASE)

    return FULLVERSION

def setup_package():
    metadata = dict(
        name = 'lightcurve_pca',
        url = 'https://github.com/astroswego/lightcurve-pca',
        description = DOCLINES[0],
        long_description = "\n".join(DOCLINES[2:]),
        version = get_version_info(),
        package_dir = {'': 'src'},
        packages = [
            'lightcurve_pca',
            'lightcurve_pca_scripts'
        ],
        entry_points = {
            'console_scripts': [
                'lightcurve_pca = lightcurve_pca_scripts.lightcurve_pca:main'
            ]
        },
        keywords = [
            'astronomy',
            'light curve',
            'principal component analysis',
            'stellar pulsation',
            'variable star'
        ],
        classifiers = [f for f in CLASSIFIERS.split('\n') if f],
        requires = [
            'numpy (>= 1.8.0)',
            'matplotlib',
            'scikit (>= 0.14.0)'
        ]
    )

    from setuptools import setup

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
