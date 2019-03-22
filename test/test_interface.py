import choice_model
import pytest


class TestInterface():
    def test_no_data(self, simple_model):
        with pytest.raises(choice_model.interface.interface.NoDataLoaded):
            choice_model.Interface(simple_model)

    def test_multinomial_logit(self, simple_multinomial_model):
        with pytest.raises(TypeError):
            choice_model.Interface(simple_multinomial_model)
