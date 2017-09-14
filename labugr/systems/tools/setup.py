from __future__ import division, print_function, absolute_import

import os
import sys
import subprocess

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    numpy_nodepr_api = dict(define_macros=[("NPY_NO_DEPRECATED_API",
                                            "NPY_1_9_API_VERSION")])

    config = Configuration('tools', parent_package, top_path)
    from labugr.systems._fortran import get_sgemv_fix

    lapack_opt = get_info('lapack_opt',notfound_action=2)
    
    if sys.platform == 'win32':
        superlu_defs = [('NO_TIMER',1)]
    else:
        superlu_defs = []
    superlu_defs.append(('USE_VENDOR_BLAS',1))

    superlu_src = join(dirname(__file__), 'SuperLU', 'SRC')

    sources = list(glob.glob(join(superlu_src, '*.c')))
    headers = list(glob.glob(join(superlu_src, '*.h')))

    config.add_library('superlu_src',
                       sources=sources,
                       macros=superlu_defs,
                       include_dirs=[superlu_src],
                       )

    # Extension
    ext_sources = ['_superlumodule.c',
                   '_superlu_utils.c',
                   '_superluobject.c']
    ext_sources += get_sgemv_fix(lapack_opt)

    config.add_extension('_superlu',
                         sources=ext_sources,
                         libraries=['superlu_src'],
                         depends=(sources + headers),
                         extra_info=lapack_opt,
                         **numpy_nodepr_api
                         )

    config.add_extension('_csparsetools',
                         sources=['_csparsetools.c'])

    def get_sparsetools_sources(ext, build_dir):
        # Defer generation of source files
        subprocess.check_call([sys.executable,
                               os.path.join(os.path.dirname(__file__),
                                            'generate_sparsetools.py'),
                               '--no-force'])
        return []

    depends = ['sparsetools_impl.h',
               'bsr_impl.h',
               'csc_impl.h',
               'csr_impl.h',
               'other_impl.h',
               'bool_ops.h',
               'bsr.h',
               'complex_ops.h',
               'coo.h',
               'csc.h',
               'csgraph.h',
               'csr.h',
               'dense.h',
               'dia.h',
               'py3k.h',
               'sparsetools.h',
               'util.h']
    depends = [os.path.join('sparsetools', hdr) for hdr in depends],
    config.add_extension('_sparsetools',
                         define_macros=[('__STDC_FORMAT_MACROS', 1)],
                         depends=depends,
                         include_dirs=['sparsetools'],
                         sources=[os.path.join('sparsetools', 'sparsetools.cxx'),
                                  os.path.join('sparsetools', 'csr.cxx'),
                                  os.path.join('sparsetools', 'csc.cxx'),
                                  os.path.join('sparsetools', 'bsr.cxx'),
                                  os.path.join('sparsetools', 'other.cxx'),
                                  get_sparsetools_sources]
                         )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
