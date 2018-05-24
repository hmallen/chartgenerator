from setuptools import setup, find_packages

setup(
    name='chartgenerator',
    version='0.1',
    author='Hunter M. Allen',
    author_email='allenhm@gmail.com',
    license='MIT',
    #packages=find_packages(),
    packages=['chartgenerator'],
    #scripts=['bin/heartbeatmonitor.py'],
    install_requires=['numpy>=1.2.1',
                      'pandas>=0.21.0',
                      'peakutils>=1.1.1',
                      'plotly>=2.6.0'],
    description='Candlestick chart generator and support/resistance calculator for market data using Plot.ly and peakutils',
    keywords=['chart', 'generator'],
)
