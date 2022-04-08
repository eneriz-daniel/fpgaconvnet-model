import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE
from fpgaconvnet.tools.resource_model import bram_memory_resource_model

@dataclass
class Queue(Module):
    depth: int

    def __post_init__(self):
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/queue_lut.npy"))
        self.rsc_coef["LUTRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/queue_lutram.npy"))

    def utilisation_model(self):
        return {
            "LUTRAM" : np.array([self.depth, self.data_width]),
            "LUT" : np.array([self.depth, 1]),
        }

    def pipeline_depth(self):
        return self.depth

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info['depth'] = self.depth
        # return the info
        return info

    def rsc(self,coef=None):
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get the linear model estimation
        rsc = Module.rsc(self, coef)
        # update with deterministic FF model
        rsc["FF"] = 2*self.int2bits(self.depth)+1
        # return the resource usage
        return rsc
