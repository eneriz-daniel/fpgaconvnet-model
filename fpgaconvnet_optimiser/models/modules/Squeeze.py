from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
<<<<<<< HEAD
=======
import sys
>>>>>>> ddfbba918a133f4b67598cc891fcb81c8985f5a9

class Squeeze(Module):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_out: int,
            coarse_in: int,
            data_width=16
        ):
        
        # module name
        self.name = "squeeze"

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # init variables
        self.coarse_out = coarse_out
        self.coarse_in  = coarse_in

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'coarse'    : self.coarse,
            'kernel_size'   : self.k_size,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b)

    def rsc(self):
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : 0,
          "DSP"  : 0,
          "FF"   : 0 #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows                           , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                           , "ERROR: invalid column dimension"
        assert data.shape[2] == int(self.channels/self.coarse_out)  , "ERROR: invalid channel dimension"
        assert data.shape[3] == self.coarse_out                     , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            int(self.channels/self.coarse_in),
            self.coarse_in),dtype=float)

        out = np.reshape(data,(self.rows,self.cols,self.channels))
        out = np.reshape(data,(self.rows,self.cols,int(self.channels/self.coarse_in),self.coarse_in))
           
        return out


