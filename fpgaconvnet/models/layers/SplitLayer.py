"""
The split/fork/branch layer.
Takes one stream input and outputs several streams using the fork module.
"""

from typing import List
import pydot
import numpy as np
import os
import math

from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.layers import MultiPortLayer

class SplitLayer(MultiPortLayer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int,
            ports_out: int = 1,
            data_width: int = 16
        ):
        """
        Parameters
        ----------
        rows: int
            row dimension of input featuremap
        cols: int
            column dimension of input featuremap
        channels: int
            channel dimension of input featuremap

        Attributes
        ----------
        buffer_depth: int, default: 0
            depth of incoming fifo buffers for each stream in.
        rows: list int
            row dimension of input featuremap
        cols: list int
            column dimension of input featuremap
        channels: list int
            channel dimension of input featuremap
        ports_in: int
            number of ports into the layer
        ports_out: int
            number of ports out of the layer
        coarse_in: list int
            number of parallel streams per port into the layer.
        coarse_out: NEED TO DEFINE
           TODO
        data_width: int
            bitwidth of featuremap pixels
        modules: dict
            dictionary of `module` instances that make
            up the layer. These modules are used for the
            resource and performance models of the layer.
        """

        # initialise parent class
        super().__init__([rows], [cols], [channels], [coarse], [coarse],
                ports_out=ports_out, data_width=data_width)

        # parameters
        self._coarse = coarse

        # init modules
        #One fork module, fork coarse_out corresponds to number of layer output ports
        self.modules["fork"] = Fork( self.rows_in(), self.cols_in(),
                self.channels_in(), 1, self.ports_out)

        # update the modules
        self.update()

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return [self._coarse]

    @property
    def coarse_out(self) -> int:
        return [self._coarse]*self.ports_out

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self.coarse_out = [val]
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self._coarse_out = [val]
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self._coarse_out = [val]
        self.update()

    def rows_out(self, port_index=0) -> int:
        return self.rows[0]

    def cols_out(self, port_index=0) -> int:
        return self.cols[0]

    def channels_out(self, port_index=0) -> int:
        return self.channels[0]

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse

    def update(self):
        # fork
        self.modules['fork'].rows     = self.rows_in()
        self.modules['fork'].cols     = self.cols_in()
        self.modules['fork'].channels = self.channels_in()//self.coarse
        self.modules['fork'].coarse   = self.ports_out

    def resource(self):

        # get module resources
        fork_rsc = self.modules['fork'].rsc()

        #Total
        return {
            "LUT"   :   fork_rsc['LUT']*self.coarse,
            "FF"    :   fork_rsc['FF']*self.coarse,
            "BRAM"  :   fork_rsc['BRAM']*self.coarse,
            "DSP"   :   fork_rsc['DSP']*self.coarse
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"split",str(i)]), label="split" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"split",str(i)]) for i in range(self.coarse) ]
        nodes_out = [ "_".join([name,"split",str(i)]) for i in range(self.ports_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"
