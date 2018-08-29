from utils.sampleinfo import SampleProperties
from utils.EventSamples import EventSample
%matplotlib inline

gen_options = {"PrintMode": False,
               "Nsignal": 100,
               "SourceLoc": [180, 40]}


dict_sample1 = {"samplename": "ANTtrack",
                "sigmaPSF": 0.5,
                "badrecorate": 0.05,
                "bginfile": "PublicData_0712.dat"}

sampleprop1 = SampleProperties(dict_sample1)

sample1 = EventSample(sampleprop1)
sample1.generate_events(gen_options["Nsignal"], gen_options["SourceLoc"])

if gen_options["PrintMode"]:
    sample1.PlotEventMap()
