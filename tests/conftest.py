import pytest

PICK = None
PATIENCE = None

def pytest_addoption(parser):
    parser.addoption('--count', action='store', default='3')
    parser.addoption('--patience', action='store', default=20)
    parser.addoption(
        '--runslow', action='store_true', default=False,
        help='Run simple network training routines.')

def pytest_configure(config):
    global PICK, PATIENCE
    PICK = int(config.getoption('--count'))
    PATIENCE = int(config.getoption('--patience'))

    config.addinivalue_line('markers', 'slow: mark a test as slow')

def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'): return
    skip_slow = pytest.mark.skip(reason='need --runslow to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
    
