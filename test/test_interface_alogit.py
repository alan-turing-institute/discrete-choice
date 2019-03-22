import choice_model
import os.path
import platform
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


class TestAlogitInterface():
    def test_multinomial_logit(self, simple_multinomial_model_with_data):
        model = simple_multinomial_model_with_data
        choice_model.AlogitInterface(model, './dummy')

    def test_alogit_path(self, simple_multinomial_model_with_data):
        model = simple_multinomial_model_with_data
        interface = choice_model.AlogitInterface(model, 'alo.exe')
        assert interface.alogit_path == os.path.abspath('alo.exe')

    def test_simple_model(self, simple_model):
        with pytest.raises(TypeError):
            choice_model.AlogitInterface(simple_model, './dummy')

    def test_no_data(self, simple_multinomial_model):
        with pytest.raises(choice_model.interface.interface.NoDataLoaded):
            choice_model.AlogitInterface(simple_multinomial_model, './dummy')


@pytest.fixture(scope="module")
def simple_multinomial_alogit_interface(simple_multinomial_model_with_data):
    return choice_model.AlogitInterface(simple_multinomial_model_with_data,
                                        './dummy')


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
        print(interface._specify_data_file())
        assert (
            interface._specify_data_file() ==
            ['file (name=Simple.csv) var1 var2 choice1_va choice2_va '
             'avail_cho1 avail_cho2', 'choice_no']
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

    def test_choices_record(self, simple_multinomial_alogit_interface):
        interface = simple_multinomial_alogit_interface
        assert interface._define_choices() == [
            'choice=recode(choice_no choice1, choice2)']

    def test_alo_file(self, simple_multinomial_model_with_data, tmp_path):
        temp = tmp_path
        data_file = temp / 'simple.csv'
        alo_file = temp / 'simple.alo'
        interface = choice_model.AlogitInterface(
            simple_multinomial_model_with_data,
            './dummy',
            data_file=str(data_file.absolute()),
            alo_file=str(alo_file.absolute())
            )
        interface._write_alo_file()
        file_text = alo_file.read_text()
        # Skip checking file name as this will be different between runs
        assert file_text[:85] == (
            '$title  Simple model\n$estimate\n$coeff p1 p2 p3 cchoice1\n$nest'
            ' root() choice1 choice2\n'
            )
        assert file_text[-267:] == (
            'Avail(choice1) = avail_cho1\nAvail(choice2) = '
            'avail_cho2\nchoice=recode(choice_no choice1, choice2)\n$array '
            'var3(alts)\nvar3(choice1) = choice1_va\nvar3(choice2) = '
            'choice2_va\nUtil(choice1) = cchoice1 + p1*var1 + p3*var3(choice1)'
            '\nUtil(choice2) = p2*var2 + p3*var3(choice2)\n'
            )


class TestDataFile():
    def test_data_file(self, simple_multinomial_model_with_data, tmp_path):
        temp = tmp_path
        data_file = temp / 'simple.csv'
        alo_file = temp / 'simple.alo'
        interface = choice_model.AlogitInterface(
            simple_multinomial_model_with_data,
            './dummy',
            data_file=str(data_file.absolute()),
            alo_file=str(alo_file.absolute())
            )
        interface._write_data_file()
        assert data_file.read_text() == '1,2,3,4,1,1,1.0\n5,6,7,8,1,1,2.0\n'


@pytest.fixture(scope="module")
def simple_multinomial_alogit_estimation(simple_multinomial_model_with_data):
    interface = choice_model.AlogitInterface(
        simple_multinomial_model_with_data,
        r'D:\Alo45.exe')
    interface.estimate()
    return interface


@pytest.mark.skip(reason="ALOGIT doesn't like this model")
@pytest.mark.skipif(platform.system() != 'Windows',
                    reason='ALOGIT only runs on Windows')
class TestAlogitEstimation():
    def test_null_log_likelihood(self, simple_multinomial_alogit_estimation):
        interface = simple_multinomial_alogit_estimation
        assert interface.null_log_likelihood() == pytest.approx(-1.3863,
                                                                1.0e-4)

    def test_final_log_likelihood(self, simple_multinomial_alogit_estimation):
        interface = simple_multinomial_alogit_estimation
        assert interface.final_log_likelihood() == pytest.approx(-1.0627e-07,
                                                                 1.0e-5)


@pytest.fixture(scope='module')
def grenoble_estimation():
    with open(main_data_dir+'grenoble.yml') as model_file,\
            open(main_data_dir+'grenoble.csv') as data_file:
        model = choice_model.MultinomialLogit.from_yaml(model_file)
        model.load_data(data_file)
    interface = choice_model.AlogitInterface(model, r'D:\Alo45.exe')
    interface.estimate()
    return interface


@pytest.mark.skipif(platform.system() != 'Windows',
                    reason='ALOGIT only runs on Windows')
class TestPylogitGrenobleEstimation():
    def test_null_log_likelihood(self, grenoble_estimation):
        interface = grenoble_estimation
        assert interface.null_log_likelihood() == pytest.approx(-1452.5186,
                                                                1.0e-4)

    def test_final_log_likelihood(self, grenoble_estimation):
        interface = grenoble_estimation
        assert interface.final_log_likelihood() == pytest.approx(-828.5038,
                                                                 1.0e-4)

    @pytest.mark.parametrize('parameter,value', [
        ('cpass', -2.731),
        ('cwalk', 2.100),
        ('ccycle', .5976),
        ('cpt', 1.098),
        ('pfemale_passenger', .8481),
        ('phas_car', 1.123),
        ('pmanual_worker', .7553),
        ('pcentral_zone', -1.481),
        ('pfemale_cycle', -.9190),
        ('pcar_competition', 2.655),
        ('porigin_walk', -.1890E-02),
        ('phead_of_household', -.8310),
        ('pnon_linear', -.3240E-02),
        ('pcost', -.1127E-02),
        ('ptime', -.3840E-03)
        ])
    def test_optimised_parameters(self, grenoble_estimation_example,
                                  parameter, value):
        interface = grenoble_estimation_example
        parameters = interface.parameters()
        assert parameters[parameter] == pytest.approx(value, rel=1.0e-3)

    @pytest.mark.parametrize('parameter,error', [
        ('cpass', .650),
        ('cwalk', .312),
        ('ccycle', .325),
        ('cpt', .365),
        ('pfemale_passenger', .343),
        ('phas_car', .501),
        ('pmanual_worker', .234),
        ('pcentral_zone', .488),
        ('pfemale_cycle', .246),
        ('pcar_competition', .342),
        ('porigin_walk', .128E-02),
        ('phead_of_household', .219),
        ('pnon_linear', .333E-03),
        ('pcost', .411E-03),
        ('ptime', .958E-04)
        ])
    def test_standard_errors(self, grenoble_estimation_example, parameter,
                             error):
        interface = grenoble_estimation_example
        errors = interface.standard_errors()
        assert errors[parameter] == pytest.approx(error, rel=1.0e-2)

    @pytest.mark.parametrize('parameter,t_value', [
        ('cpass', -4.2),
        ('cwalk', 6.7),
        ('ccycle', 1.8),
        ('cpt', 3.0),
        ('pfemale_passenger', 2.5),
        ('phas_car', 2.2),
        ('pmanual_worker', 3.2),
        ('pcentral_zone', -3.0),
        ('pfemale_cycle', -3.7),
        ('pcar_competition', 7.8),
        ('porigin_walk', -1.5),
        ('phead_of_household', -3.8),
        ('pnon_linear', -9.7),
        ('pcost', -2.7),
        ('ptime', -4.0)
        ])
    def test_t_values(self, grenoble_estimation_example, parameter, t_value):
        interface = grenoble_estimation_example
        t_values = interface.t_values()
        assert t_values[parameter] == pytest.approx(t_value, rel=1.0e-2)

    def test_estimation_time(self, grenoble_estimation_example):
        interface = grenoble_estimation_example
        assert interface.estimation_time() > 0.0


@pytest.fixture(scope='module')
def grenoble_estimation_example():
    with open(main_data_dir+'grenoble.yml') as model_file,\
            open(main_data_dir+'grenoble.csv') as data_file:
        model = choice_model.MultinomialLogit.from_yaml(model_file)
        model.load_data(data_file)
    interface = choice_model.AlogitInterface(model, './dummy')
    interface._parse_output_file(data_dir+'Grenoble.LOG')
    interface._estimated = True
    return interface


class TestPylogitGrenobleEstimationExample():
    def test_null_log_likelihood(self, grenoble_estimation_example):
        interface = grenoble_estimation_example
        assert interface.null_log_likelihood() == -1452.5186

    def test_final_log_likelihood(self, grenoble_estimation_example):
        interface = grenoble_estimation_example
        assert interface.final_log_likelihood() == -828.5038

    @pytest.mark.parametrize('parameter,value', [
        ('cpass', -2.731),
        ('cwalk', 2.100),
        ('ccycle', .5976),
        ('cpt', 1.098),
        ('pfemale_passenger', .8481),
        ('phas_car', 1.123),
        ('pmanual_worker', .7553),
        ('pcentral_zone', -1.481),
        ('pfemale_cycle', -.9190),
        ('pcar_competition', 2.655),
        ('porigin_walk', -.1890E-02),
        ('phead_of_household', -.8310),
        ('pnon_linear', -.3240E-02),
        ('pcost', -.1127E-02),
        ('ptime', -.3840E-03)
        ])
    def test_optimised_parameters(self, grenoble_estimation_example,
                                  parameter, value):
        interface = grenoble_estimation_example
        parameters = interface.parameters()
        assert parameters[parameter] == value

    @pytest.mark.parametrize('parameter,error', [
        ('cpass', .650),
        ('cwalk', .312),
        ('ccycle', .325),
        ('cpt', .365),
        ('pfemale_passenger', .343),
        ('phas_car', .501),
        ('pmanual_worker', .234),
        ('pcentral_zone', .488),
        ('pfemale_cycle', .246),
        ('pcar_competition', .342),
        ('porigin_walk', .128E-02),
        ('phead_of_household', .219),
        ('pnon_linear', .333E-03),
        ('pcost', .411E-03),
        ('ptime', .958E-04)
        ])
    def test_standard_errors(self, grenoble_estimation_example, parameter,
                             error):
        interface = grenoble_estimation_example
        errors = interface.standard_errors()
        assert errors[parameter] == error

    @pytest.mark.parametrize('parameter,t_value', [
        ('cpass', -4.2),
        ('cwalk', 6.7),
        ('ccycle', 1.8),
        ('cpt', 3.0),
        ('pfemale_passenger', 2.5),
        ('phas_car', 2.2),
        ('pmanual_worker', 3.2),
        ('pcentral_zone', -3.0),
        ('pfemale_cycle', -3.7),
        ('pcar_competition', 7.8),
        ('porigin_walk', -1.5),
        ('phead_of_household', -3.8),
        ('pnon_linear', -9.7),
        ('pcost', -2.7),
        ('ptime', -4.0)
        ])
    def test_t_values(self, grenoble_estimation_example, parameter, t_value):
        interface = grenoble_estimation_example
        t_values = interface.t_values()
        assert t_values[parameter] == t_value

    def test_estimation_time(self, grenoble_estimation_example):
        interface = grenoble_estimation_example
        assert interface.estimation_time() == 0.03


class TestAlogitRequiresEstimation():
    @pytest.mark.parametrize('method', [
        'display_results',
        'null_log_likelihood',
        'final_log_likelihood',
        'parameters',
        'standard_errors',
        't_values',
        'estimation_time'
        ])
    def test_requires_estimation(self, simple_multinomial_alogit_interface,
                                 method):
        interface = simple_multinomial_alogit_interface
        with pytest.raises(choice_model.interface.interface.NotEstimated):
            getattr(interface, method)()
