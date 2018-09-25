from utils.sampleinfo import SampleProperties
from utils.EventSamples import EventSample
from utils.ClusterAnalyser import ClusterAnalyser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
# %matplotlib inline

# %%
def generate_and_analyse(gen_options, dict_sample):

    sampleprop1 = SampleProperties(dict_sample)
    sample1 = EventSample(sampleprop1)
    OutputDataFrame = pd.DataFrame({"TS": [], "ns": [], "LikMax": [],
                                    "Lbg": []})

    for iEvt in range(gen_options["Nsimulations"]):

        sample1.generate_events(gen_options["Nsignal"], gen_options["SourceLoc"])

        cluster1 = ClusterAnalyser(gen_options["SourceLoc"], coneangle=5,
                                   sample=sample1)
        opt_values = cluster1.TScalculation(iEvt)

        OutputDataFrame = OutputDataFrame.append(pd.DataFrame(opt_values))
        sample1.ClearSample()

    outfilename = "{0}/TS_N{1}_dec{2}.h5".format(gen_options["outfiledir"],
                                                 gen_options["Nsignal"],
                                                 gen_options["SourceLoc"][1])

    OutputDataFrame.to_hdf(outfilename, key="DF", mode="w")


# %%

if __name__=="__main__":

    dict_sample1 = {"samplename": "ANTtrack",
                    "sigmaPSF": 0.5,
                    "badrecorate": 0.05,
                    "bginfile": "PublicData_0712.dat"}

    p = mp.Pool(4)

    list_opts = []
    list_sampinfo = []

    for Nsource in range(31):

        Nsimulations = 10000
        if Nsource == 0:
            Nsimulations *= 10
        gen_options = {"PrintMode": False,
                       "Nsimulations": Nsimulations,
                       "Nsignal": Nsource,
                       "SourceLoc": [180, -45],
                       "outfiledir": "."}

        list_opts.append(gen_options)
        list_sampinfo.append(dict_sample1)
        #generate_and_analyse(gen_options, dict_sample1)

    p.starmap(generate_and_analyse, zip(list_opts,list_sampinfo))
    #generate_and_analyse(gen_options, dict_sample1)
    p.close()
    p.join()
