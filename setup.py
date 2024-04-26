from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'House Price Prediction'
LONG_DESCRIPTION = 'Housign Price Prediction Module'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="housingPricePrediction",
    version=1.0,
    author="Danasekarane Arounachalam",
    author_email="danas@tiger.com",
    description="Simple Package",
    long_description="Long Package",
    packages=find_packages(),
    install_requires=['bs4', 'numpy', 'pandas', 'scikit-learn'],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'non standard code'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
