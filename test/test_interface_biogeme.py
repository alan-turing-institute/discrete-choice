import choice_model
import pytest


@pytest.fixture(scope='module')
def simple_multinomial_biogeme_interface(simple_multinomial_model_with_data):
    return choice_model.BiogemeInterface(simple_multinomial_model_with_data)


class TestBiogemeInterface():
    def test_multinomial_logit(self, simple_multinomial_model_with_data):
        choice_model.BiogemeInterface(simple_multinomial_model_with_data)

    def test_simple_model(self, simple_model):
        with pytest.raises(TypeError):
            choice_model.BiogemeInterface(simple_model)

    def test_no_data(self, simple_multinomial_model):
        with pytest.raises(choice_model.interface.interface.NoDataLoaded):
            choice_model.BiogemeInterface(simple_multinomial_model)

# Tests for constructing utilities


class TestPylogitRequiresEstimation():
    @pytest.mark.parametrize('method', [
        'display_results',
        'null_log_likelihood',
        'final_log_likelihood',
        'parameters',
        'standard_errors',
        't_values',
        'estimation_time'
        ])
    def test_requires_estimation(self, simple_multinomial_biogeme_interface,
                                 method):
        interface = simple_multinomial_biogeme_interface
        with pytest.raises(choice_model.interface.interface.NotEstimated):
            getattr(interface, method)()
