"""
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

@dataclass
class AveragePool(Module):
    backend: str = "chisel"

    def __post_init__(self):
        return

    def utilisation_model(self):#TODO - copied from acum, FIXME
        return {
            "LUT"   : np.array([1]),
            "FF"    : np.array([1]),
            "DSP"   : np.array([1]),
            "BRAM"  : np.array([1]),
        }

    def rows_out(self):
        return 1

    def cols_out(self):
        return 1

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # return the info
        return info

    def rsc(self,coef=None):
        # get the linear model estimation
        # rsc = Module.rsc(self, coef)
        # # return the resource model
        # return rsc
        return {
            "LUT"   : 1,
            "FF"    : 1,
            "DSP"   : 0,
            "BRAM"  : 0,
        }

    def visualise(self, name):
        return pydot.Node(name, label="average_pool", shape="box",
                style="filled", fillcolor="chartreuse",
                fontsize=MODULE_FONTSIZE)


    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels   , "ERROR: invalid channel dimension"

        # return average
        return np.average(data, axis=(0,1))


