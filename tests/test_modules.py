import glob
import unittest
import ddt
import json
from fpgaconvnet_optimiser.models.modules import *


class TestModuleTemplate():

    def run_test_methods_exist(self, module):
        self.assertTrue(hasattr(module, "rows_in"))
        self.assertTrue(hasattr(module, "cols_in"))
        self.assertTrue(hasattr(module, "channels_in"))
        self.assertTrue(hasattr(module, "rows_out"))
        self.assertTrue(hasattr(module, "cols_out"))
        self.assertTrue(hasattr(module, "channels_out"))

    def run_test_dimensions(self, module):
        # check input dimensions
        self.assertGreater(module.rows_in(), 0)
        self.assertGreater(module.cols_in(), 0)
        self.assertGreater(module.channels_in(), 0)
        # check output dimensions
        self.assertGreater(module.rows_out(), 0)
        self.assertGreater(module.cols_out(), 0)
        self.assertGreater(module.channels_out(), 0)

    def run_test_rates(self, module):
        # check rate in
        self.assertGreaterEqual(module.rate_in(), 0.0)
        self.assertLessEqual(module.rate_in(),1.0)
        # check rate out
        self.assertGreaterEqual(module.rate_out(), 0.0)
        self.assertLessEqual(module.rate_out(), 1.0)

@ddt.ddt
class TestForkModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/fork/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Fork([config["channels"],config["rows"],config["cols"]],
                config["kernel_size"],
                config["coarse"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

@ddt.ddt
class TestAccumModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/accum/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Accum([config["channels"],config["rows"],config["cols"]],
                config["filters"],config["groups"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

        # additional checks
        self.assertGreater(module.filters,0)

@ddt.ddt
class TestConvModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/conv/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Conv([config["channels"],config["rows"],config["cols"]],
                config["filters"],config["fine"],config["kernel_size"],config["group"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

@ddt.ddt
class TestGlueModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/glue/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Glue([config["channels"],config["rows"],config["cols"]],
                config["filters"],config["coarse_in"],config["coarse_out"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

@ddt.ddt
class TestSlidingWindowModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/sliding_window/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = SlidingWindow([config["channels"],config["rows"],config["cols"]],
                config["kernel_size"],config["stride"],config["pad_top"],
                config["pad_right"],config["pad_bottom"],config["pad_left"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

@ddt.ddt
class TestPoolModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/pool/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Pool([config["channels"],config["rows"],config["cols"]],
                config["kernel_size"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

@ddt.ddt
class TestSqueezeModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/squeeze/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Squeeze([config["channels"],config["rows"],config["cols"]],
                config["coarse_in"],config["coarse_out"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

@ddt.ddt
class TestReLUModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/modules/relu/*.json"))
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ReLU([config["channels"],config["rows"],config["cols"]])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)

