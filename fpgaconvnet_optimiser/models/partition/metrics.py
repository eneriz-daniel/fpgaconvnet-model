import numpy as np
import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.matrix as matrix

def get_pipeline_depth(self, node): # TODO: change to longest path problem
    # find the pipeline depth of the current node
    pipeline_depth = self.graph.nodes[node]['hw'].pipeline_depth()
    # find the longest path to end from this node
    if self.graph.out_degree(node) == 0:
        return pipeline_depth
    else:
        return pipeline_depth + max([ 
            self.get_pipeline_depth(edge) for edge in graphs.get_next_nodes(self.graph,node) ])

def get_interval(self):
    # get the interval matrix        
    interval_matrix = matrix.get_interval_matrix(self.graph)
    # return the overall interval
    return np.max(np.absolute(interval_matrix))

def get_latency(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    # get pipeline depth of partition
    input_node = graphs.get_input_nodes(self.graph)[0]
    pipeline_depth = self.get_pipeline_depth(input_node) # TODO: find max of all input nodes
    # return the latency (in seconds)
    batch_size  = int(self.batch_size)
    wr_factor   = self.wr_factor
    size_wr     = self.size_wr
    return ( (interval*batch_size+pipeline_depth)*wr_factor + (wr_factor-1)*size_wr )/(freq*1000000) 

def get_bandwidth_in(self,freq):
    # get the slowest rate
    rates_matrix = matrix.get_rates_matrix(self.graph)
    rate = np.max(np.absolute(rates_matrix))
    # get workload of input
    input_node = graphs.get_input_nodes(self.graph)[0]
    workload_in = self.graph.nodes[input_node]["hw"].workload_in(0)
    # get bandwidth (GB/s)
    return (workload_in*self.streams_in*freq)/(rate*1000)

def get_bandwidth_out(self,freq):
    # get the slowest rate
    rates_matrix = matrix.get_rates_matrix(self.graph)
    rate = np.max(np.absolute(rates_matrix))
    # get workload of input
    output_node = graphs.get_output_nodes(self.graph)[0]
    workload_out = self.graph.nodes[output_node]["hw"].workload_out(0)
    # get bandwidth (GB/s)
    return (workload_out*self.streams_out*freq)/(rate*1000)

def get_total_operations(self):
    return sum([self.graph.nodes[node]['hw'].get_operations() for node in self.graph.nodes])
    
