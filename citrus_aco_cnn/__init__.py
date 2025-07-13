"""Top-level package for citrus_aco_cnn."""

__all__ = []

# Convenient re-export of the main entry point
from importlib import import_module as _imp

def _main():
    _imp("citrus_aco_cnn.cli").main()

# Allow `python -m citrus_aco_cnn`
if __name__ == "__main__":
    _main()
