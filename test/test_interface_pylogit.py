from .context import add_project_path, data_dir
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
