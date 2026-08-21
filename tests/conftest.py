"""Shared fixtures for all test modules."""
import pytest
from backend.middleware import reset_limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the in-memory rate limiter before every test to avoid cross-module429s."""
    reset_limiter()
    yield
