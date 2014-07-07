from os import makedirs
from os.path import isdir
from multiprocessing import Pool
from numpy import resize

__all__ = [
    'pmap',
    'make_sure_path_exists',
    'colvec',
    'rowvec'
]

def pmap(func, args, processes=None, callback=lambda *x: None, **kwargs):
    if processes is 1:
        results = []
        for arg in args:
            result = func(arg, **kwargs)
            results.append(result)
            callback(result)
        return results
    else:
        p = Pool() if processes is None else Pool(processes)
        results = [p.apply_async(func, (arg,), kwargs, callback)
                   for arg in args]
        p.close()
        p.join()
        return [result.get() for result in results]
    
def make_sure_path_exists(path):
    """Creates the supplied path. Raises OS error if the path cannot be
    created."""
    try:
      makedirs(path)
    except OSError:
      if not isdir(path):
        raise

def colvec(X):
    return resize(X, (X.shape[0], 1))

def rowvec(X):
    return resize(X, (1, X.shape[0]))[0]
