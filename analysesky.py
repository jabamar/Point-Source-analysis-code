from utils.sampleinfo import SampleProperties
from utils.EventSamples import EventSample
from utils.ClusterAnalyser import ClusterAnalyser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
%matplotlib inline

gen_options = {"PrintMode": False,
               "Nsignal": 10,
               "SourceLoc": [180, -45],
               "outfiledir": "."}


dict_sample1 = {"samplename": "ANTtrack",
                "sigmaPSF": 0.5,
                "badrecorate": 0.05,
                "bginfile": "PublicData_0712.dat"}

sampleprop1 = SampleProperties(dict_sample1)

sample1 = EventSample(sampleprop1)
# %%
%%time

OutputDataFrame = pd.DataFrame({"TS": [], "ns": [], "LikMax": [], "Lbg": []})

for iEvt in range(1000):

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
print(OutputDataFrame.head())
plt.hist(OutputDataFrame["TS"], bins=100, range=[-1,20])
plt.yscale("log")
