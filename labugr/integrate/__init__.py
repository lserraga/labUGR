from .quadpack import quad, dblquad, tplquad, nquad

excluded = ['excluded', 'quadpack']

__all__ = [s for s in dir() if not ((s in excluded)or s.startswith('_'))]

from labugr.testing.utils import PytestTester
test = PytestTester(__name__)
del PytestTester