"""
Unit and regression test for the aludel package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import aludel


def test_aludel_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "aludel" in sys.modules
