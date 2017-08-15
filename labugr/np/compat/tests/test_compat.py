from __future__ import division, absolute_import, print_function

from os.path import join

from labugr.np.compat import isfileobj
from labugr.np.testing import assert_, run_module_suite
from labugr.np.testing import tempdir


def test_isfileobj():
    with tempdir(prefix="numpy_test_compat_") as folder:
        filename = join(folder, 'a.bin')

        with open(filename, 'wb') as f:
            assert_(isfileobj(f))

        with open(filename, 'ab') as f:
            assert_(isfileobj(f))

        with open(filename, 'rb') as f:
            assert_(isfileobj(f))


if __name__ == "__main__":
    run_module_suite()
