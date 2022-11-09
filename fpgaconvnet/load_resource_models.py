import os
from fpgaconvnet.tools.resource_regression_model import ModuleModel

# CHISEL_MODULES = [ "Accum", "AveragePool", "Bias", "Fork", "Glue",
CHISEL_MODULES = [ "Accum", "Bias", "Fork", "Glue", "SlidingWindow", "Squeeze", "VectorDot" ] # averagepool, pool, relu
CHISEL_MODULES = [ "MaxPool" ] # averagepool, pool, relu
HLS_MODULES = []

# iterate over modules
for module in CHISEL_MODULES:

    print(f"{module} (chisel)")
    # create regression model
    m = ModuleModel(module, "chisel")

    # load data
    m.load_data_from_db()

    # fit model
    m.fit_model(from_cache=False)

    # save coefficeints
    cache_path = os.path.join(
            os.path.dirname(__file__),
            f"coefficients/chisel")
    m.save_coef(cache_path)
