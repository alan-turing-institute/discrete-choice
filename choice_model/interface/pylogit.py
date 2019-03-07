"""
pylogit interface
"""

from . import Interface
from .. import MultinomialLogit


class PylogitInterface(Interface):
    _valid_models = [MultinomialLogit]

    def __init__(self, model):
        super().__init__(model)
