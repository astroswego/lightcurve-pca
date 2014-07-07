import numpy
from math import modf

__all__ = [
    'rephase',
    'get_phase'
]

def rephase(data, period=1, col=0):
    rephased = numpy.ma.copy(data)
    rephased.T[col] = [get_phase(x[col], period)
                       for x in rephased]
    return rephased

def get_phase(time, period=1, offset=0):
    return (modf(time/period)[0]-offset)%1
