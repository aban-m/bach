import pytest

PICK = None

def pytest_addoption(parser):
    parser.addoption('--count', action='store', default='10')

def pytest_configure(config):
    global PICK
    PICK = int(config.getoption('--count'))

def fun_with_flags(): return 'Indeed.'
