"""
Stream learning module for River and CapyMOA integration.
"""

from .river_wrapper import KAFRegressor, KAFClassifier
from .capymoa_wrapper import CapyMOARegressor

__all__ = ['KAFRegressor', 'KAFClassifier', 'CapyMOARegressor']
