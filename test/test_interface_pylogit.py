import choice_model
import pytest


@pytest.fixture(scope='module')
def simple_multinomial_model():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.MultinomialLogit.from_yaml(yaml_file)


@pytest.fixture(scope="module")
def simple_model():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file:
        return choice_model.ChoiceModel.from_yaml(yaml_file)


@pytest.fixture(scope="module")
def simple_multinomial_model_with_data():
    with open(data_dir+'simple_model.yml', 'r') as yaml_file,\
            open(data_dir+'simple.csv', 'r') as data_file:
        model = choice_model.MultinomialLogit.from_yaml(yaml_file)
        model.load_data(data_file)
        return model


@pytest.fixture(scope="module")
def simple_multinomial_pylogit_interface(simple_multinomial_model_with_data):
    return choice_model.PylogitInterface(simple_multinomial_model_with_data)


class TestPylogitInterface():
    def test_multinomial_logit(self, simple_multinomial_model_with_data):
        interface = choice_model.PylogitInterface(
                simple_multinomial_model_with_data)
        assert interface.model == simple_multinomial_model_with_data

    def test_simple_model(self, simple_model):
        with pytest.raises(TypeError):
            choice_model.PylogitInterface(simple_model)

    def test_no_data(self, simple_multinomial_model):
        with pytest.raises(choice_model.interface.interface.NoDataLoaded):
            choice_model.PylogitInterface(simple_multinomial_model)


class TestPylogitLongData():
    @pytest.mark.parametrize(
        'wide_data_key,long_data_key,observation,choice', [
            ('var1', 'var1', 0, 0),
            ('var1', 'var1', 0, 1),
            ('var2', 'var2', 0, 0),
            ('var2', 'var2', 0, 1),
            ('var1', 'var1', 1, 0),
            ('var1', 'var1', 1, 1),
            ('var2', 'var2', 1, 0),
            ('var2', 'var2', 1, 1),
            ('choice1_var3', 'var3', 0, 0),
            ('choice2_var3', 'var3', 0, 1),
            ('choice1_var3', 'var3', 1, 0),
            ('choice2_var3', 'var3', 1, 1)
            ]
        )
    def test_variables(self, simple_multinomial_pylogit_interface,
                       wide_data_key, long_data_key, observation,
                       choice):
        interface = simple_multinomial_pylogit_interface
        wide_data = interface.model.data
        long_data = interface.long_data
        wide_value = wide_data[wide_data_key][observation]
        long_value = long_data[long_data_key][observation*2+choice]
        assert wide_value == long_value

    @pytest.mark.parametrize('observation', [0, 1])
    def test_choices(self, simple_multinomial_pylogit_interface, observation):
        interface = simple_multinomial_pylogit_interface
        wide_data = interface.model.data
        long_data = interface.long_data
        if wide_data['alternative'][observation] == 'choice1':
            assert long_data['choice_bool'][observation*2] == 1
        elif wide_data['alternative'][observation] == 'choice2':
            assert long_data['choice_bool'][observation*2+1] == 1
        else:
            assert 0


class TestPylogitSpecification():
    def test_intercepts(self, simple_multinomial_pylogit_interface):
        interface = simple_multinomial_pylogit_interface
        assert interface.specification['intercept'] == [1]

    @pytest.mark.parametrize('variable,specification', [
        ('var1', [1]),
        ('var2', [2]),
        ('var3', [1, 2])
        ])
    def test_variables(self, simple_multinomial_pylogit_interface,
                       variable, specification):
        interface = simple_multinomial_pylogit_interface
        assert interface.specification[variable] == [specification]


class TestPylogitNames():
    def test_intercepts(self, simple_multinomial_pylogit_interface):
        interface = simple_multinomial_pylogit_interface
        assert interface.names['intercept'] == ['cchoice1']

    @pytest.mark.parametrize('variable,names', [
        ('var1', 'p1'),
        ('var2', 'p2'),
        ('var3', 'p3')
        ])
    def test_variables(self, simple_multinomial_pylogit_interface,
                       variable, names):
        interface = simple_multinomial_pylogit_interface
        assert interface.names[variable] == [names]


@pytest.fixture(scope="module")
def simple_multinomial_pylogit_estimation(simple_multinomial_model_with_data):
    interface = choice_model.PylogitInterface(
        simple_multinomial_model_with_data)
    interface.estimate()
    return interface


class TestPylogitEstimation():
    @pytest.mark.parametrize('attribute,value', [
        ('null_log_likelihood', -1.38629),
        ('log_likelihood', -1.0627e-07),
        ('rho_squared', 1),
        ('rho_bar_squared', -1.88539)
        ])
    def test_estimation(self, simple_multinomial_pylogit_estimation,
                        attribute, value):
        interface = simple_multinomial_pylogit_estimation
        assert (interface.pylogit_model.__getattribute__(attribute)
                == pytest.approx(value, 1.0e-5))

    @pytest.mark.parametrize('parameter,value', [
        ('cchoice1', 12.318752),
        ('p1', -10.433292),
        ('p2', -1.885460),
        ('p3', -12.318752)
        ])
    def test_optimised_paramters(self, simple_multinomial_pylogit_estimation,
                                 parameter, value):
        interface = simple_multinomial_pylogit_estimation
        parameters = interface.parameters()
        assert parameters[parameter] == pytest.approx(value)

    def test_null_log_likelihood(self, simple_multinomial_pylogit_estimation):
        interface = simple_multinomial_pylogit_estimation
        assert interface.null_log_likelihood() == pytest.approx(-1.38629,
                                                                1.0e-5)

    def test_final_log_likelihood(self, simple_multinomial_pylogit_estimation):
        interface = simple_multinomial_pylogit_estimation
        assert interface.final_log_likelihood() == pytest.approx(-1.0627e-07,
                                                                 1.0e-5)


@pytest.fixture(scope='module')
def grenoble_estimation():
    with open(main_data_dir+'grenoble.yml') as model_file,\
            open(main_data_dir+'grenoble.csv') as data_file:
        model = choice_model.MultinomialLogit.from_yaml(model_file)
        model.load_data(data_file)
    interface = choice_model.PylogitInterface(model)
    interface.estimate()
    return interface


class TestPylogitGrenobleEstimation():
    @pytest.mark.parametrize('attribute,value', [
        ('null_log_likelihood', -1452.5185654443776),
        ('log_likelihood', -828.503745607559),
        ('rho_squared', 0.42960884265593535),
        ('rho_bar_squared', 0.41928195227611365)
        ])
    def test_estimation(self, grenoble_estimation,
                        attribute, value):
        interface = grenoble_estimation
        assert (interface.pylogit_model.__getattribute__(attribute)
                == pytest.approx(value, 1.0e-5))

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
        assert parameters[parameter] == pytest.approx(value, rel=1.0e-3)

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
    def test_requires_estimation(self, simple_multinomial_pylogit_interface,
                                 method):
        interface = simple_multinomial_pylogit_interface
        with pytest.raises(choice_model.interface.interface.NotEstimated):
            getattr(interface, method)()
