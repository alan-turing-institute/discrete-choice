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
        assert parameters[parameter] == pytest.approx(value, rel=1.6e-3)

    @pytest.mark.parametrize('parameter,error', [
        ('cpt', 0.390846),
        ('ccycle', 0.323384),
        ('cwalk', 0.314923),
        ('cpass', 0.571899),
        ('phead_of_household', 0.236262),
        ('porigin_walk', 0.001276),
        ('pcar_competition', 0.350013),
        ('phas_car', 0.468693),
        ('pfemale_passenger', 0.333083),
        ('pfemale_cycle', 0.231491),
        ('pcentral_zone', 0.461484),
        ('pmanual_worker', 0.219562),
        ('ptime', 0.000111),
        ('pcost', 0.000402),
        ('pnon_linear', 0.000313)
        ])
    def test_standard_errors(self, grenoble_estimation, parameter, error):
        interface = grenoble_estimation
        errors = interface.standard_errors()
        assert errors[parameter] == pytest.approx(error, rel=1.0e-2)

    @pytest.mark.parametrize('parameter,t_value', [
        ('cpt', 2.809782),
        ('ccycle', 1.847984),
        ('cwalk', 6.666838),
        ('cpass', -4.774690),
        ('phead_of_household', -3.517141),
        ('porigin_walk', -1.481077),
        ('pcar_competition', 7.584170),
        ('phas_car', 2.395480),
        ('pfemale_passenger', 2.546299),
        ('pfemale_cycle', -3.970099),
        ('pcentral_zone', -3.209405),
        ('pmanual_worker', 3.440241),
        ('ptime', -3.463463),
        ('pcost', -2.804146),
        ('pnon_linear', -10.350534),
        ])
    def test_t_values(self, grenoble_estimation, parameter, t_value):
        interface = grenoble_estimation
        t_values = interface.t_values()
        assert t_values[parameter] == pytest.approx(t_value, rel=1.0e-2)

    def test_estimation_time(self, grenoble_estimation):
        interface = grenoble_estimation
        assert interface.estimation_time() > 0.0


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
