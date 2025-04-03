import unittest
import pkgutil
import importlib


def load_tests(loader, tests, pattern):
    """Automatically discover and load all test modules."""
    suite = unittest.TestSuite()
    
    for package in ["myapp.tests.unit", "myapp.tests.integration", "myapp.tests.e2e"]:
        for _, module_name, _ in pkgutil.iter_modules([package.replace(".", "/")]):
            module = importlib.import_module(f"{package}.{module_name}")
            suite.addTests(loader.loadTestsFromModule(module))

    return suite
