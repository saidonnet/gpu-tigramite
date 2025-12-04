"""
GPU-Tigramite Setup Script

Modern build system using CMake and pybind11 for CUDA acceleration.
Supports Python 3.10-3.12 and CUDA 11.x-12.x.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Custom extension that uses CMake to build"""
    
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command that runs CMake"""
    
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Force GCC 10 for CUDA 11.8 compatibility (GCC 12+ not supported)
        gcc10_path = '/usr/bin/gcc-10'
        gxx10_path = '/usr/bin/g++-10'
        
        if os.path.exists(gcc10_path) and os.path.exists(gxx10_path):
            print(f"✓ Using GCC 10 for CUDA compatibility")
            os.environ['CC'] = gcc10_path
            os.environ['CXX'] = gxx10_path
            os.environ['CUDAHOSTCXX'] = gxx10_path
        else:
            print("⚠ WARNING: GCC 10 not found! CUDA compilation may fail with GCC 12+")
            print("  Install with: sudo apt-get install gcc-10 g++-10")
        
        # CMake configuration
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={"Debug" if self.debug else "Release"}',
        ]
        
        # Explicitly set C/C++ compilers for CMake if GCC 10 is available
        if os.path.exists(gcc10_path):
            cmake_args.extend([
                f'-DCMAKE_C_COMPILER={gcc10_path}',
                f'-DCMAKE_CXX_COMPILER={gxx10_path}',
            ])
        
        # Set CUDA compiler for CMake
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        if os.path.exists(nvcc_path):
            cmake_args.append(f'-DCMAKE_CUDA_COMPILER={nvcc_path}')
        else:
            print(f"⚠ WARNING: nvcc not found at {nvcc_path}")
            print("  CUDA build will likely fail")
        
        # Force CUDA architecture for double-precision atomicAdd support
        # Requires compute capability >= 6.0 (Pascal+)
        # Default to 70 (V100) for modern GPUs, can be overridden via env var
        cuda_arch = os.environ.get('CMAKE_CUDA_ARCHITECTURES', '70')
        cmake_args.append(f'-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}')
        print(f"✓ Targeting CUDA architecture: compute_{cuda_arch} (double atomics require ≥60)")
        
        # Platform-specific configuration
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        if platform.system() == "Windows":
            cmake_args += [
                f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}',
            ]
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']
        
        # Environment variables for CMake
        env = os.environ.copy()
        
        # Set CUDA compiler path for CMake
        cuda_home = env.get('CUDA_HOME', '/usr/local/cuda')
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        if os.path.exists(nvcc_path):
            env['CUDACXX'] = nvcc_path
            env['CMAKE_CUDA_COMPILER'] = nvcc_path
        else:
            print(f"⚠ WARNING: nvcc not found at {nvcc_path}")
            print("  Set CUDA_HOME environment variable or ensure CUDA is in /usr/local/cuda")
        
        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # Run CMake configuration
        print(f"Running CMake in {build_temp}")
        print(f"CMake args: {cmake_args}")
        
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=build_temp,
            env=env
        )
        
        # Build
        print(f"Building with CMake")
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_temp
        )
        
        # For editable installs, copy the built .so to source tree
        # This is necessary because editable installs import from src/ directory
        import glob
        import shutil
        built_so = glob.glob(str(build_temp / '**' / 'gpucmiknn*.so'), recursive=True)
        if built_so and hasattr(self, 'editable_mode') and self.editable_mode:
            target_dir = Path(ext.sourcedir) / 'src' / 'gpu_tigramite' / 'cuda'
            target_dir.mkdir(parents=True, exist_ok=True)
            for so_file in built_so:
                shutil.copy2(so_file, target_dir)
                print(f"✓ Copied {Path(so_file).name} to {target_dir} for editable install")


# Read long description from README
here = Path(__file__).parent.resolve()
readme_path = here / 'README.md'
requirements_path = here / 'requirements.txt'

if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'GPU-accelerated Conditional Mutual Information for Tigramite'

# Read requirements
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = ['numpy>=1.20.0', 'torch>=2.0.0', 'tigramite>=5.0.0']

setup(
    name='gpu-tigramite',
    version='1.0.0',
    author='Saïd RAHMANI',
    author_email='saidonnet@gmail.com',
    description='GPU-accelerated Conditional Mutual Information for Tigramite (50-430x faster)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/saidonnet/gpu-tigramite',
    project_urls={
        'Documentation': 'https://gpu-tigramite.readthedocs.io',
        'Source': 'https://github.com/saidonnet/gpu-tigramite',
        'Tracker': 'https://github.com/saidonnet/gpu-tigramite/issues',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    # CRITICAL: Extension module must be built to gpu_tigramite/cuda/ subdirectory
    ext_modules=[CMakeExtension('gpu_tigramite.cuda.gpucmiknn')],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    zip_safe=False,
    include_package_data=True,
    # Ensure CUDA module subdirectory is included
    package_data={'gpu_tigramite': ['cuda/*.so', 'cuda/*.pyd']},
)