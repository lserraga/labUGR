
from .quadpack import quad, dblquad, tplquad, nquad

from labugr.testing.utils import PytestTester
test = PytestTester(__name__)
del PytestTester