from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='chartgenerator',
    version='0.1a19',
    author='Hunter M. Allen',
    author_email='allenhm@gmail.com',
    license='MIT',
    #packages=find_packages(),
    packages=['chartgenerator'],
    #scripts=['bin/heartbeatmonitor.py'],
    install_requires=['numpy',
                      'numpyencoder',
                      'pandas',
                      'peakutils',
                      'plotly'],
    description='Candlestick chart generator and support/resistance calculator for market data using Plot.ly and peakutils.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hmallen/chartgenerator',
    keywords=['chart', 'generator'],
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
)
