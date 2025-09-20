"""Configuration package placeholder."""

# For this scaffold we keep configuration simple and environment-driven.
import os


def get(key: str, default=None):
    return os.environ.get(key, default)
