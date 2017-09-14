from __future__ import division, print_function, absolute_import

import os
import sys
import subprocess

import re
import glob
from distutils.dep_util import newer

def uses_veclib(info):
    if sys.platform != "darwin":
        return False
    r_accelerate = re.compile("vecLib")
    extra_link_args = info.get('extra_link_args', '')
    for arg in extra_link_args:
        if r_accelerate.search(arg):
            return True
    return False


def uses_accelerate(info):
    if sys.platform != "darwin":
        return False
    r_accelerate = re.compile("Accelerate")
    extra_link_args = info.get('extra_link_args', '')
    for arg in extra_link_args:
        if r_accelerate.search(arg):
            return True
    return False


def uses_mkl(info):
    r_mkl = re.compile("mkl")
    libraries = info.get('libraries', '')
    for library in libraries:
        if r_mkl.search(library):
            return True

    return False


def needs_g77_abi_wrapper(info):
    """Returns True if g77 ABI wrapper must be used."""
    if uses_accelerate(info) or uses_veclib(info):
        return True
    elif uses_mkl(info):
        return True
    else:
        return False


def get_g77_abi_wrappers(info):
    """
    Returns file names of source files containing Fortran ABI wrapper
    routines.
    """
    wrapper_sources = []

    path = os.path.abspath(os.path.dirname(__file__))
    if needs_g77_abi_wrapper(info):
        wrapper_sources += [
            os.path.join(path, 'src', 'wrap_g77_abi_f.f'),
            os.path.join(path, 'src', 'wrap_g77_abi_c.c'),
        ]
        if uses_accelerate(info):
            wrapper_sources += [
                    os.path.join(path, 'src', 'wrap_accelerate_c.c'),
                    os.path.join(path, 'src', 'wrap_accelerate_f.f'),
            ]
        elif uses_mkl(info):
            wrapper_sources += [
                    os.path.join(path, 'src', 'wrap_dummy_accelerate.f'),
            ]
        else:
            raise NotImplementedError("Do not know how to handle LAPACK %s on mac os x" % (info,))
    else:
        wrapper_sources += [
            os.path.join(path, 'src', 'wrap_dummy_g77_abi.f'),
            os.path.join(path, 'src', 'wrap_dummy_accelerate.f'),
        ]
    return wrapper_sources


def needs_sgemv_fix(info):
    """Returns True if SGEMV must be fixed."""
    if uses_accelerate(info):
        return True
    else:
        return False


def get_sgemv_fix(info):
    """ Returns source file needed to correct SGEMV """
    path = os.path.abspath(os.path.dirname(__file__))
    if needs_sgemv_fix(info):
        return [os.path.join(path, 'src', 'apple_sgemv_fix.c')]
    else:
        return []


def split_fortran_files(source_dir, subroutines=None):
    """Split each file in `source_dir` into separate files per subroutine.

    Parameters
    ----------
    source_dir : str
        Full path to directory in which sources to be split are located.
    subroutines : list of str, optional
        Subroutines to split. (Default: all)

    Returns
    -------
    fnames : list of str
        List of file names (not including any path) that were created
        in `source_dir`.

    Notes
    -----
    This function is useful for code that can't be compiled with g77 because of
    type casting errors which do work with gfortran.

    Created files are named: ``original_name + '_subr_i' + '.f'``, with ``i``
    starting at zero and ending at ``num_subroutines_in_file - 1``.

    """

    if subroutines is not None:
        subroutines = [x.lower() for x in subroutines]

    def split_file(fname):
        with open(fname, 'rb') as f:
            lines = f.readlines()
            subs = []
            need_split_next = True

            # find lines with SUBROUTINE statements
            for ix, line in enumerate(lines):
                m = re.match(b'^\\s+subroutine\\s+([a-z0-9_]+)\\s*\\(', line, re.I)
                if m and line[0] not in b'Cc!*':
                    if subroutines is not None:
                        subr_name = m.group(1).decode('ascii').lower()
                        subr_wanted = (subr_name in subroutines)
                    else:
                        subr_wanted = True
                    if subr_wanted or need_split_next:
                        need_split_next = subr_wanted
                        subs.append(ix)

            # check if no split needed
            if len(subs) <= 1:
                return [fname]

            # write out one file per subroutine
            new_fnames = []
            num_files = len(subs)
            for nfile in range(num_files):
                new_fname = fname[:-2] + '_subr_' + str(nfile) + '.f'
                new_fnames.append(new_fname)
                if not newer(fname, new_fname):
                    continue
                with open(new_fname, 'wb') as fn:
                    if nfile + 1 == num_files:
                        fn.writelines(lines[subs[nfile]:])
                    else:
                        fn.writelines(lines[subs[nfile]:subs[nfile+1]])

        return new_fnames

    exclude_pattern = re.compile('_subr_[0-9]')
    source_fnames = [f for f in glob.glob(os.path.join(source_dir, '*.f'))
                             if not exclude_pattern.search(os.path.basename(f))]
    fnames = []
    for source_fname in source_fnames:
        created_files = split_file(source_fname)
        if created_files is not None:
            for cfile in created_files:
                fnames.append(os.path.basename(cfile))

    return fnames


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    from os.path import join, dirname
    numpy_nodepr_api = dict(define_macros=[("NPY_NO_DEPRECATED_API",
                                            "NPY_1_9_API_VERSION")])

    config = Configuration('tools', parent_package, top_path)

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
        # combinatorics
    config.add_extension('_comb',
                         sources=['_comb.c'])


    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
