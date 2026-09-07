# src/utils.py
"""
Utility functions for saving, loading, and project management.
"""

import os
import time
import joblib
import numpy as np
from functools import wraps


def save_model(model, filepath):
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"💾 Model saved: {filepath}")


def load_model(filepath):
    """Load a trained model from disk."""
    model = joblib.load(filepath)
    print(f"📂 Model loaded: {filepath}")
    return model


def save_numpy(array, filepath):
    """Save a numpy array to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, array)
    print(f"💾 Array saved: {filepath} | Shape: {array.shape}")


def load_numpy(filepath):
    """Load a numpy array from disk."""
    array = np.load(filepath, allow_pickle=True)
    print(f"📂 Array loaded: {filepath} | Shape: {array.shape}")
    return array


def timer(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__} completed in {elapsed:.2f} seconds")
        return result
    return wrapper


def print_header(title):
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title):
    """Print a formatted sub-section header."""
    print(f"\n--- {title} ---")
