import glob
import unittest
import ddt
import json
from fpgaconvnet.models.layers import *

class TestLayer3DTemplate():

    def run_test_dimensions(self, layer):
        # check input dimensions
        self.assertTrue(layer.rows_in() > 0)
        self.assertTrue(layer.cols_in() > 0)
        self.assertTrue(layer.depth_in() > 0)
        self.assertTrue(layer.channels_in() > 0)
        # check output dimensions
        self.assertTrue(layer.rows_out() > 0)
        self.assertTrue(layer.cols_out() > 0)
        self.assertTrue(layer.depth_out() > 0)
        self.assertTrue(layer.channels_out() > 0)

    def run_test_rates(self, layer):
        # check rate in
        self.assertTrue(layer.rate_in() >= 0.0)
        self.assertTrue(layer.rate_in() <= 1.0)
        # check rate out
        self.assertTrue(layer.rate_out() >= 0.0)
        self.assertTrue(layer.rate_out() <= 1.0)

    def run_test_workload(self,layer):
        # check workload in
        self.assertTrue(layer.workload_in() >= 0.0)
        # check workload out
        self.assertTrue(layer.workload_out() >= 0.0)

    def run_test_streams(self,layer):
        # check streams in
        self.assertTrue(layer.streams_in() >= 1)
        # check streams out
        self.assertTrue(layer.streams_out() >= 1)

    def run_test_size(self,layer):
        # check size in
        self.assertTrue(layer.size_in() >= 1)
        # check size out
        self.assertTrue(layer.size_out() >= 1)

    def run_test_latency(self,layer):
        # check latency in
        self.assertTrue(layer.latency_in() >= 0.0)
        # check latency out
        self.assertTrue(layer.latency_out() >= 0.0)

    def run_test_pipeline_depth(self,layer):
        # check pipeline depth
        self.assertTrue(layer.pipeline_depth() >= 0.0)

    def run_test_wait_depth(self,layer):
        # check wait depth
        #self.assertTrue(layer.wait_depth() >= 0.0)
        pass

    def run_test_resources(self,layer):
        # check resources
        rsc = layer.resource()
        self.assertEqual(set(list(rsc.keys())), set(["BRAM","DSP","LUT","FF"]))
        self.assertTrue(rsc["LUT"] >= 0)
        self.assertTrue(rsc["FF"] >= 0)
        self.assertTrue(rsc["DSP"] >= 0)
        self.assertTrue(rsc["BRAM"] >= 0)

    def run_test_updating_properties(self, layer):
        # updating coarse in
        coarse_in = max(layer.get_coarse_in_feasible())
        layer.coarse_in = coarse_in
        self.assertEqual(layer.coarse_in, coarse_in)
        # updating coarse out
        coarse_out = max(layer.get_coarse_out_feasible())
        layer.coarse_out = coarse_out
        self.assertEqual(layer.coarse_out, coarse_out)


@ddt.ddt
class TestPoolingLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/pooling3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = PoolingLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            coarse=config["coarse"],
            kernel_rows=config["kernel_rows"],
            kernel_cols=config["kernel_cols"],
            kernel_depth=config["kernel_depth"],
            stride_rows=config["stride_rows"],
            stride_cols=config["stride_cols"],
            stride_depth=config["stride_depth"],
            pad_top=config["pad_top"],
            pad_right=config["pad_right"],
            pad_front=config["pad_front"],
            pad_bottom=config["pad_bottom"],
            pad_left=config["pad_left"],
            pad_back=config["pad_back"],
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

# @ddt.ddt
# class TestConvolutionLayer(TestLayer3DTemplate,unittest.TestCase):

#     @ddt.data(
#         "tests/configs/layers/convolution/config_0.json",
#         "tests/configs/layers/convolution/config_1.json",
#         "tests/configs/layers/convolution/config_2.json",
#         "tests/configs/layers/convolution/config_3.json",
#         "tests/configs/layers/convolution/config_4.json",
#         "tests/configs/layers/convolution/config_7.json",
#         "tests/configs/layers/convolution/config_8.json",
#         "tests/configs/layers/convolution/config_9.json",
#         "tests/configs/layers/convolution/config_10.json",
#         "tests/configs/layers/convolution/config_11.json",
#         "tests/configs/layers/convolution/config_12.json",
#         "tests/configs/layers/convolution/config_13.json",
#         "tests/configs/layers/convolution/config_14.json",
#         "tests/configs/layers/convolution/config_15.json",
#         "tests/configs/layers/convolution/config_16.json",
#         "tests/configs/layers/convolution/config_17.json",
#         "tests/configs/layers/convolution/config_18.json",
#         "tests/configs/layers/convolution/config_19.json",
#          "tests/configs/layers/convolution/config_23.json",
#          "tests/configs/layers/convolution/config_25.json",
#     )
#     def test_layer_configurations(self, config_path):

#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise layer
#         layer = ConvolutionLayer(
#             config["filters"],
#             config["rows"],
#             config["cols"],
#             config["channels"],
#             coarse_in=config["coarse_in"],
#             coarse_out=config["coarse_out"],
#             kernel_size=config["kernel_size"],
#             stride=config["stride"],
#             groups=config["groups"],
#             pad=config["pad"],
#             fine=config["fine"],
#             has_bias=config["has_bias"]
#         )

#         # run tests
#         self.run_test_dimensions(layer)
#         self.run_test_rates(layer)
#         self.run_test_workload(layer)
#         self.run_test_size(layer)
#         self.run_test_streams(layer)
#         self.run_test_latency(layer)
#         self.run_test_pipeline_depth(layer)
#         self.run_test_wait_depth(layer)
#         self.run_test_updating_properties(layer)
#         self.run_test_resources(layer)

@ddt.ddt
class TestActivationLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/activation3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ActivationLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            config["activation_type"],
            coarse = config["coarse"]
        )
        print(f"Type {config['activation_type']}: latency {layer.latency()} cycles, resources: {layer.resource()}. Coarse: {config['coarse']}")
        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

@ddt.ddt
class TestReLULayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/relu3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ReLULayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            coarse = config["coarse"]
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

# @ddt.ddt
# class TestInnerProductLayer(TestLayer3DTemplate,unittest.TestCase):

#     @ddt.data(
#         "tests/configs/layers/inner_product/config_0.json",
#         "tests/configs/layers/inner_product/config_1.json",
#         "tests/configs/layers/inner_product/config_2.json",
#         "tests/configs/layers/inner_product/config_3.json",
#         "tests/configs/layers/inner_product/config_4.json",
#         "tests/configs/layers/inner_product/config_5.json",
#         "tests/configs/layers/inner_product/config_6.json",
#         "tests/configs/layers/inner_product/config_7.json",
#         "tests/configs/layers/inner_product/config_8.json",
#         "tests/configs/layers/inner_product/config_9.json",
#         "tests/configs/layers/inner_product/config_10.json",
#     )
#     def test_layer_configurations(self, config_path):

#         # open configuration
#         with open(config_path, "r") as f:
#             config = json.load(f)

#         # initialise layer
#         layer = InnerProductLayer(
#             config["filters"],
#             config["rows"],
#             config["cols"],
#             config["channels"],
#             config["coarse_in"],
#             config["coarse_out"],
#             has_bias=config["has_bias"]
#         )

#         # run tests
#         self.run_test_dimensions(layer)
#         self.run_test_rates(layer)
#         self.run_test_workload(layer)
#         self.run_test_size(layer)
#         self.run_test_streams(layer)
#         self.run_test_latency(layer)
#         self.run_test_pipeline_depth(layer)
#         self.run_test_wait_depth(layer)
#         self.run_test_updating_properties(layer)
#         self.run_test_resources(layer)

@ddt.ddt
class TestSqueezeLayer3D(TestLayer3DTemplate,unittest.TestCase):

    @ddt.data(*glob.glob("tests/configs/layers/squeeze3d/*.json"))
    def test_layer_configurations(self, config_path):

        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = SqueezeLayer3D(
            config["rows"],
            config["cols"],
            config["depth"],
            config["channels"],
            config["coarse_in"],
            config["coarse_out"],
        )

        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)
        self.run_test_workload(layer)
        self.run_test_size(layer)
        self.run_test_streams(layer)
        self.run_test_latency(layer)
        self.run_test_pipeline_depth(layer)
        self.run_test_wait_depth(layer)
        self.run_test_updating_properties(layer)
        self.run_test_resources(layer)

