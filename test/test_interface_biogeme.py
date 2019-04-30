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


@pytest.fixture(scope='module')
def grenoble_estimation(main_data_dir):
    with open(main_data_dir+'grenoble.yml') as model_file,\
            open(main_data_dir+'grenoble.csv') as data_file:
        model = choice_model.MultinomialLogit.from_yaml(model_file)
        model.load_data(data_file)
    interface = choice_model.BiogemeInterface(model)
    interface.estimate()
    return interface


class TestBiogemeGrenobleEstimation():
    @pytest.mark.parametrize('parameter,value', [
        ('cpt', 1.098191),
        ('ccycle', 0.597608),
        ('cwalk', 2.099543),
        ('cpass', -2.730642),
        ('phead_of_household', -0.830965),
        ('porigin_walk', -0.001890),
        ('pcar_competition', 2.654556),
        ('phas_car', 1.122745),
        ('pfemale_passenger', 0.848129),
        ('pfemale_cycle', -0.919043),
        ('pcentral_zone', -1.481088),
        ('pmanual_worker', 0.755348),
        ('ptime', -0.000384),
        ('pcost', -0.001127),
        ('pnon_linear', -0.003240)
        ])
    def test_optimised_parameters(self, grenoble_estimation,
                                  parameter, value):
        interface = grenoble_estimation
        parameters = interface.parameters()
        assert parameters[parameter] == pytest.approx(value, rel=2.0e-3)


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
