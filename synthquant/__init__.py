"""SynthQuant: Production-grade synthetic financial data generation and probabilistic forecasting.

See https://github.com/AndrewFSee/synthquant for full documentation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synthquant")
except PackageNotFoundError:
    __version__ = "0.0.0"

from synthquant.config import Settings, get_settings

__all__ = ["__version__", "Settings", "get_settings"]
