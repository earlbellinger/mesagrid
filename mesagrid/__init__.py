"""Top-level package for mesagrid."""

from importlib.metadata import version
__version__ = version("mesagrid")

__author__ = """Earl Patrick Bellinger"""
__email__ = 'earl.bellinger@yale.edu'

from mesagrid.star import Track, Grid, load_history_seismology
