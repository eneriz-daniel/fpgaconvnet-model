"""
The Glue module is used to combine streams
used for channel parallelism in the
Convolution layer together.

.. figure:: ../../../figures/glue_diagram.png
"""
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pydot

from fpgaconvnet.models.modules import Module, MODULE_FONTSIZE

@dataclass
class Glue(Module):
    filters: int
    coarse_in: int
    coarse_out: int
    acc_width: int = field(default=16, init=False)
    backend: str = "chisel"

    def channels_in(self):
        return self.filters

    def channels_out(self):
        return self.filters

    def get_latency(self):
        return self.rows *self.cols *self.filters / self.coarse_out

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["filters"] = self.filters
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

    def utilisation_model(self):
        if self.backend == "hls":
            pass
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([1]),
                "LUT_RAM"   : np.array([1]),
                "LUT_SR"    : np.array([1]),
                "FF"        : np.array([1]),
                "DSP"       : np.array([0]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def visualise(self, name):
        return pydot.Node(name,label="glue", shape="box",
                style="filled", fillcolor="fuchsia",
                fontsize=MODULE_FONTSIZE)

    def functional_model(self,data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == int(self.filters/self.coarse_out) , "ERROR: invalid  dimension"
        assert data.shape[3] == self.coarse_in , "ERROR: invalid  dimension"
        assert data.shape[4] == self.coarse_out , "ERROR: invalid  dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            int(self.filters/self.coarse_out),
            self.coarse_out),dtype=float)

        for index,_ in np.ndenumerate(out):
            for c in range(self.coarse_in):
                out[index] += data[index[0],index[1],index[2],c,index[3]]

        return out

