
from setuptools import setup, find_packages

setup(
    name="tickerBot",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tickerbot = tickerbot.main:main',
        ],
    },
    install_requires=[
        'yfinance',
        'stocknews',
        'pandas',
        'get-all-tickers',
        'numpy',
        'requests',
        'multitasking',
        'platformdirs',
        'pytz',
        'frozendict',
        'peewee',
        'beautifulsoup4',
        'curl_cffi',
        'protobuf',
        'websockets',
        'feedparser',
        'nltk',
        'tradingview-screener',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A stock trading bot with suggestion and automatic modes.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/tickerBot",
)
