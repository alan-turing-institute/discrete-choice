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


class TestAlogitInterface():
    def test_multinomial_logit(self, simple_multinomial_model_with_data):
        model = simple_multinomial_model_with_data
        interface = choice_model.AlogitInterface(model)
        interface._write_alo_file()

    def test_simple_model(self, simple_model):
        with pytest.raises(TypeError):
            choice_model.AlogitInterface(simple_model)

    def test_no_data(self, simple_multinomial_model):
        with pytest.raises(choice_model.interface.interface.NoDataLoaded):
            choice_model.AlogitInterface(simple_multinomial_model)


@pytest.fixture(scope="module")
def simple_multinomial_alogit_interface(simple_multinomial_model_with_data):
    return choice_model.AlogitInterface(simple_multinomial_model_with_data)


class TestAbbreviation():
    abbreviation_map = [
        ('choice1', 'choice1'),
        ('choice2', 'choice2'),
        ('alternative', 'alternativ'),
        ('avail_choice1', 'avail_cho1'),
        ('avail_choice2', 'avail_cho2'),
        ('var1', 'var1'),
        ('var2', 'var2'),
        ('var3', 'var3'),
        ('choice1_var3', 'choice1_va'),
        ('choice2_var3', 'choice2_va'),
        ('cchoice1', 'cchoice1'),
        ('p1', 'p1'),
        ('p2', 'p2'),
        ('p3', 'p3')
        ]

    @pytest.mark.parametrize('full,short', abbreviation_map)
    def test_abbreviation(self, simple_multinomial_alogit_interface,
                          full, short):
        interface = simple_multinomial_alogit_interface
        assert interface.abbreviate(full) == short

    @pytest.mark.parametrize('full,short', abbreviation_map)
    def test_elongation(self, simple_multinomial_alogit_interface,
                        full, short):
        interface = simple_multinomial_alogit_interface
        assert interface.elongate(short) == full


class TestAloFile():
    @pytest.mark.parametrize('choice,string', [
        ('choice1', 'cchoice1 + p1*var1 + p3*var3(choice1)'),
        ('choice2', 'p2*var2 + p3*var3(choice2)')
        ])
    def test_utility_string(self, simple_multinomial_alogit_interface, choice,
                            string):
        interface = simple_multinomial_alogit_interface
        assert interface._utility_string(choice) == string

    def test_data_file_string(self, simple_multinomial_alogit_interface):
        interface = simple_multinomial_alogit_interface
        assert (
            interface._specify_data_file('file.alo') ==
            ['file (name=file.alo) var1 var2 choice1_va choice2_va avail_cho1'
             ' avail_cho2', 'alternativ']
            )

    @pytest.mark.parametrize('array,argument,string', [
        ('var3', 'choice1', 'var3(choice1)'),
        ('var3', 'choice2', 'var3(choice2)')
        ])
    def test_array(self, simple_multinomial_alogit_interface, array, argument,
                   string):
        interface = simple_multinomial_alogit_interface
        assert interface._array(array, argument) == string

    @pytest.mark.parametrize('array,argument,string', [
        ('var3', 'choice1', 'var3(choice1) ='),
        ('var3', 'choice2', 'var3(choice2) =')
        ])
    def test_array_record(self, simple_multinomial_alogit_interface, array,
                          argument, string):
        interface = simple_multinomial_alogit_interface
        assert interface._array_record(array, argument) == string
