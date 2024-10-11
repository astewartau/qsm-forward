from setuptools import setup, find_packages

setup(
    name='qsm-forward',
    version='0.23',
    packages=find_packages(),
    url='https://github.com/astewartau/qsm-forward',
    author='Ashley Stewart',
    author_email='a.stewart.au@gmail.com',
    description='A forward-model simulation for Quantitative Susceptibility Mapping',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'nilearn',
        'dipy'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': ['qsm-forward=qsm_forward.main:main'],
    },
)

