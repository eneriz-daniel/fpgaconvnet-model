import os
import json
import numpy as np
import importlib
import inspect
import dataclasses
from collections import namedtuple
from tqdm import tqdm

import sklearn.linear_model
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from fpgaconvnet.models.modules import Module

HLS_RSC_TYPES=["LUT", "FF", "DSP", "BRAM"]
CHISEL_RSC_TYPES=["Logic_LUT", "LUT_RAM", "LUT_SR", "FF", "DSP", "BRAM36", "BRAM18"]

SERVER_DB="mongodb+srv://fpgaconvnet.hwnxpyo.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"

class ModuleModel:
    def __init__(self, identifier, module, regression_model, backend):

        self.identifier = identifier
        self.module = module
        self.backend = backend
        assert regression_model in ["linear_regression", "xgboost"], f"regression model {regression_model} not supported"
        self.regression_model = regression_model

        if self.backend == "chisel":
            self.rsc_types = CHISEL_RSC_TYPES
        elif self.backend == "hls":
            self.rsc_types = HLS_RSC_TYPES

        self.parameters = []
        self.model = {k: [] for k in self.rsc_types}
        self.coef = {k: [] for k in self.rsc_types}
        self.predict = {k: [] for k in self.rsc_types}
        self.actual = {k: [] for k in self.rsc_types}

    def load_data_from_db(self):
        """
        loads a set of parameter-resource pairs for
        many different configurations and runs of the
        module. This is to be used with the
        `fit_resource_model` method.
        """
        # database .pem path
        db_pem = os.path.join(os.path.dirname(__file__),
                "fpgaconvnet-mongodb.pem")

        # create MongoDB client
        client = MongoClient(SERVER_DB, tls=True,
                tlsCertificateKeyFile=db_pem,
                server_api=ServerApi('1'))

        # load database collection for given backend
        db = client["fpgaconvnet"]
        collection = db[self.backend]

        if self.backend == "chisel":
            filter = {
                "name" : self.identifier, # specific module
                "parameters.data_width": 16, # only 16-bit input shapes
                "time_stamp.commit_hash": "08bf16da9441d36e01d9cf1021e7602bf9e738fd", # specific commit hash
                # filter resources
                "resources.LUT" : { "$lt" : int(461000*0.8) },
                "resources.FF"  : { "$lt" : int(504000*0.8) },
                "resources.DSP" : { "$lt" : int(1728*0.8) },
                "resources.BRAM36" : { "$lt" : int(312*0.8) },
                "resources.BRAM18" : { "$lt" : int(624*0.8) },
            }
        else:
            raise NotImplementedError

        for document in tqdm(collection.find(filter), desc="loading points from database"):
            self.parameters.append(document["parameters"])
            for rsc in self.rsc_types:
                self.actual[rsc].append(document["resources"][rsc])

        assert len(self.parameters) > 0

    def load_data_from_cache(self, filepath):
        folders = [f for f in os.listdir(filepath) if not f]
        for folder in folders:
            filename = os.path.join(filepath, folder, "results.json")
            try:
                with open(filename,"r") as jsonfile:
                    oneDict = json.load(jsonfile)
                    self.parameters.append(oneDict["parameters"])
                    for rsc in self.rsc_types:
                        self.actual[rsc].append(oneDict["resources"][rsc])
            except:
                pass

        assert len(self.parameters) > 0

    def fit_model(self, from_cache=False):

        if from_cache:

            # get the cache path
            match self.regression_model:
                case "linear_regression":
                    cache_path = os.path.dirname(__file__) + f"../coefficients/{self.regression_model}/{self.backend}"

                    # iterate over resource types
                    self.coef = {}
                    for rsc_type in self.rsc_types:
                        with open(f"{cache_path}/{self.identifier}_{rsc_type}.npy", "wb") as f:
                            self.coef[rsc_type] = np.save(f)
                case "xgboost":
                    cache_path = os.path.dirname(__file__) + f"../coefficients/{self.regression_model}/{self.backend}"

                    # iterate over resource types
                    self.coef = {}
                    for rsc_type in self.rsc_types:
                        model = XGBRegressor()
                        model.load_model(f"{cache_path}/{self.identifier}_{rsc_type}.json")
                        self.coef[rsc_type] = model

        else:

            # get the utilisation model
            utilisation_model = getattr(importlib.import_module(
                f"fpgaconvnet.models.modules.{self.module}"),
                self.module).utilisation_model
            utilisation_model = staticmethod(utilisation_model)

            # iterate over design points
            for point in self.parameters:
                point["backend"] = self.backend
                point_obj = namedtuple('ParameterPoint', point.keys())(*point.values())
                for rsc_type in self.rsc_types:
                    match self.regression_model:
                        case "linear_regression":
                            self.model[rsc_type].append(utilisation_model(point_obj)[rsc_type])
                        case "xgboost":
                            tmp = list(point.values())
                            tmp.remove('chisel')
                            final_tmp = []
                            for i in tmp:
                                if type(i) == list:
                                    for j in i:
                                        final_tmp.append(j)
                                elif type(i) == str:
                                    continue
                                else:
                                    final_tmp.append(i)
                            self.model[rsc_type].append(np.array(final_tmp))
            # fit coefficients
            for rsc_type in self.rsc_types:
                match self.regression_model:
                    case "linear_regression":
                        self.coef[rsc_type] = self.get_nnls_coef(
                                np.array(self.model[rsc_type]),
                                np.array(self.actual[rsc_type]))
                    case "xgboost":
                        self.coef[rsc_type] = self.get_nnls_model(
                                np.array(self.model[rsc_type]),
                                np.array(self.actual[rsc_type]))

            # iterate over design points
            # for point in self.parameters:
            #     point["backend"] = self.backend
            #     point_obj = namedtuple('ParameterPoint', point.keys())(*point.values())
            #     for rsc_type in self.rsc_types:
            #         self.model[rsc_type].append(utilisation_model(point_obj)[rsc_type])

    def get_nnls_coef(self, model, rsc):
        """
        a method for fitting a regression model using
        non-negative least squares. This ensures all
        coefficients are positive.
        """

        # fit model
        nnls = sklearn.linear_model.LinearRegression(
                positive=True, fit_intercept=False)
        # nnls = sklearn.linear_model.LinearRegression(
        #         positive=False, fit_intercept=False)

        # return coefficients
        return nnls.fit(model, rsc).coef_

    def get_nnls_model(self, model, rsc):
        """
        a method for fitting a regression model using
        non-negative least squares. This ensures all
        coefficients are positive.
        """

        # initialize model
        nnls = XGBRegressor()

        # fit model
        nnls.fit(model, rsc)

        return nnls

    def save_coef(self, outpath):
        for rsc_type in self.rsc_types:
            match self.regression_model:
                case "linear_regression":
                    filepath = os.path.join(outpath, f"{self.module}_{rsc_type}.npy".lower())
                    np.save(filepath, self.coef[rsc_type])
                case "xgboost":
                    filepath = os.path.join(outpath, f"{self.module}_{rsc_type}.json".lower())
                    self.coef[rsc_type].save_model(filepath)

    # def get_model_error(self):
    #     for rsc_type in self.rsc_types:
    #         diff = np.array(self.predict[rsc_type]) - np.array(self.actual[rsc_type])
    #         mean_err = diff.mean()
    #         mse = (diff*diff).mean()
    #         std = diff.std()

    #         print("="*5 + "{}_model".format(rsc_type) + "="*5)
    #         print("Base: mean_err={0}, mse={1}, std={2}".format(mean_err, mse, std))
    #         print("\n")

    # def plot_results(self, outpath):
    #     for rsc_type in self.rsc_types:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         x = self.actual[rsc_type]
    #         y = self.predict[rsc_type]
    #         ax.plot(x, x, label="actual")
    #         ax.scatter(x, y, label="predict", marker="x", color="r")
    #         ax.set_title(rsc_type)
    #         ax.set_xlabel("actual")
    #         ax.set_ylabel("predicted")
    #         ax.legend()

    #         filepath = os.path.join(outpath, f"{self.name}_{rsc_type}.jpg".lower())
    #         fig.savefig(filepath)

