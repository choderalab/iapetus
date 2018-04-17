"""
Unit and regression test for the iapetus package.
"""

# Import package, test suite, and other packages as needed
import iapetus
import pytest
import sys

def test_iapetus_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "iapetus" in sys.modules
