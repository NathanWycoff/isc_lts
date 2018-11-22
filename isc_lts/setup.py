from setuptools import setup

setup(
    name='ISCLatentTimeSeries',
    url='https://github.com/NathanWycoff/isc_lts',
    author='Nathan Wycoff',
    author_email='nathw95@vt.edu',
    # Needed to actually package something
    packages=['isc_lts'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='GPL',
    description='Intersubject Correlations for, say fMRI data via the Latent Time Series model.',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
