import numpy as np
import math
import os
from dataclasses import dataclass, field

from fpgaconvnet.models.modules import int2bits, Module
from fpgaconvnet.tools.resource_analytical_model import dsp_multiplier_resource_model

@dataclass
class VectorDot(Module):
    filters: int
    fine: int

    def rate_in(self):
        return 1.0/float(self.filters)

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'fine'    : self.fine,
            'filters'    : self.filters,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def utilisation_model(self):
        if self.backend == "hls":
            pass
        elif self.backend == "chisel":
            return {
                "Logic_LUT" : np.array([
                    self.fine, self.data_width,
                ]),
                "LUT_RAM"   : np.array([0]),
                "LUT_SR"    : np.array([0]),
                "FF"    : np.array([
                    int2bits(self.filters),
                ]),
                "DSP"       : np.array([self.fine]),
                "BRAM36"    : np.array([0]),
                "BRAM18"    : np.array([0]),
            }
        else:
            raise ValueError(f"{self.backend} backend not supported")

    def rsc(self,coef=None):

        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef

        # get the linear model estimation
        rsc = Module.rsc(self, coef)

        # get the dsp usage
        dsp = self.fine*dsp_multiplier_resource_model(
                self.data_width, self.data_width)
        rsc["DSP"] = dsp

        return rsc

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data, weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.fine    , "ERROR: invalid column dimension"
        # check input dimensionality
        assert weights.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert weights.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert weights.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[3] == self.filters , "ERROR: invalid channel dimension"
        assert weights.shape[4] == self.fine    , "ERROR: invalid column dimension"


        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels,
            self.filters),dtype=object)

        for index,_ in np.ndenumerate(weights):
            if index[4] == 0:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3]
                ] = weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[4]
                    ]
            else:
                out[
                    index[0],
                    index[1],
                    index[2],
                    index[3]
                ] += weights[index] * data[
                        index[0],
                        index[1],
                        index[2],
                        index[4]
                    ]

        return out


