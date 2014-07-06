from os import makedirs
from os.path import isdir

__all__ = [
    'make_sure_path_exists'
]

def make_sure_path_exists(path):
    """Creates the supplied path. Raises OS error if the path cannot be
    created."""
    try:
      makedirs(path)
    except OSError:
      if not isdir(path):
        raise
