from setuptools import setup, find_packages

def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md') as f:
        return f.read()
    

setup(
    # Module name (lowercase)
    name='afmsims',

    # Version
    version='1.0',

    description='A package to analyse AFM via FEM simulations',

    long_description=get_readme(),

    license='MIT license',

    # Author 
    author='J. Giblin-Burnham',
    author_email='j.giblin-burnham@hotmail.com',

    # Website
    url='https://abaqus-afm-simulations.readthedocs.io/en/latest/index.html',

    # Packages to include
    packages=find_packages(include=('afmsims', 'afmsims.*')),

    # List of dependencies
    install_requires = ["numpy", "matplotlib", "scipy", "absl-py","biopython","keras","mendeleev",
                        "nglview","py3dmol","pyabaqus","pypdf","pypdf2","scp","paramiko","sphinx_rtd_theme","furo",],

    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
            # Nice theme for docs
            'sphinx_rtd_theme',
            'sphinx_autopackagesummary',
            'furo',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
        ],
    }
    )
