from .context import add_project_path, data_dir, main_data_dir
import choice_model
import pytest

add_project_path()


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
        with pytest.raises(choice_model.interface.pylogit.NoDataLoaded):
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
        parameters = interface.pylogit_model.params
        assert parameters[parameter] == pytest.approx(value)


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
        ('pnon-linear', -0.003240)
        ])
    def test_optimised_paramters(self, grenoble_estimation,
                                 parameter, value):
        interface = grenoble_estimation
        parameters = interface.pylogit_model.params
        assert parameters[parameter] == pytest.approx(value, rel=1.0e-3)