import numpy as np
import pandas as pd
from scipy.optimize import minimize
# from astropy.coordinates import SkyCoord
# import astropy.coordinates.angles as ang # import AngularSeparation
# from astropy import units as u
from .EventSamples import EventSample
# from .sampleinfo import SampleProperties
import math
# from numba import jit, jitclass, float64


# spec = [
#    ('ra', float64),
#    ('dec', float64),
#    ('sinra', float64),
#    ('sindec', float64),
#    ('cosra', float64),
#    ('cosdec', float64),
#    ('vec', float64[:]),
# ]

# @jitclass(spec)
class Event:
    """Class which considers an event.

    Attributes
    ----------

    ra: float
    right ascention (in radians)

    dec: float
    declination (in radians)

    vec: numpy array of shape (1, 3)
    vector in cartesian coordinates assuming equatorial system and  r = 1
    """

    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec
        sinra = math.sin(ra)
        sindec = math.sin(dec)
        cosra = math.cos(ra)
        cosdec = math.cos(dec)
        self.vec = np.array([cosra*cosdec, sinra*cosdec, sindec])

    def distance_event(self, ev):
        """
        Returns the angular distance to an event (in radians)

        Parameters
        ----------

        ev: Event
        event to which we want to calculate a distance
        """

        # return math.acos(np.dot(self.vec, ev.vec))
        return math.acos(self.vec[0]*ev.vec[0]
                         + self.vec[1]*ev.vec[1] + self.vec[2]*ev.vec[2])


# Discarded function: SkyCoord is just too slow!
# def Separation(SourceCoord, ra, decl):
#    EventCoord = SkyCoord(ra=ra, dec=decl, unit=u.radian)
#    sep = SourceCoord.separation(EventCoord)
#    return sep.radian


# @jit
def Separation2(Source, ra, decl):
    """
    Returns the angular distance to an event (in radians)

    Parameters
    ----------

    ra: float
    right ascension of a given event

    decl: float
    declination of a given event
    """
    newevent = Event(ra, decl)
    return Source.distance_event(newevent)


# @jit
def SeparationList(Source, ralist, declist):
    Nevts = len(ralist)
    distlist = np.zeros(Nevts)
    for i in range(Nevts):
        ra = ralist[i]
        dec = declist[i]
        dist = Source.distance_event(Event(ra, dec))
        distlist[i] = dist
    return distlist


class ClusterAnalyser:
    """Class to calculate the Test Statistic of a given cluster.

    Attributes
    ----------

    sourceloc: numpy array of shape (1, 2)
    location of the source, [ra, dec] (in equatorial coordinates)

    coneangle: float
    angular distance in which we consider Signal PSF != 0

    sample: EventSample
    Input event sample

    Neventstot: int
    Number of events in the sample

    Nin: int
    Number of events inside coneangle

    Nout: int
    Number of events outside coneangle

    df_cluster: DataFrame
    DataFrame with information of the events inside coneangle
    """

    def __init__(self, sourceloc=[180, 45], coneangle=5, sample=EventSample()):
        self.sourceloc = np.deg2rad(sourceloc)
        self.coneangle = np.deg2rad(coneangle)
        self.sample = sample
        self.Neventstot = -1
        self.Nin = -1
        self.Nout = -1
        self.df_cluster = self.EventSelection(sample.df_events)

    def EventSelection(self, dfinput=pd.DataFrame()):
        """Function which selects the events inside coneangle. It returns
        a DataFrame with the relevant information.

        Parameters
        ----------

        dfinput: DataFrame
        Input dataframe with all the events in the sample
        """

        self.Neventstot = dfinput.shape[0]

        if self.Neventstot == 0:
            raise RuntimeError("FATAL: No events inside input DataFrame!")

        # Avoid the calculation of the distance for the events whose declination
        # from the source is further away than coneangle (improves performance)
        newdf = dfinput[np.abs(dfinput["DECL_EVT"] - self.sourceloc[1])
                        < self.coneangle].copy()
        # SourceCoord = SkyCoord(ra=self.sourceloc[0], dec=self.sourceloc[1],
        #                       unit=u.radian)

        SourceCoord = Event(self.sourceloc[0], self.sourceloc[1])
        #newdf["ang_separation"] = SeparationList(SourceCoord,
        #                                         newdf["RA_EVT"].values.tolist(),
        #                                         newdf["DECL_EVT"].values.tolist())
        newdf["ang_separation"] = np.vectorize(Separation2)(SourceCoord,
                                                           newdf["RA_EVT"],
                                                           newdf["DECL_EVT"])

        newdf = newdf[newdf["ang_separation"] < self.coneangle].copy()

        self.Nin = newdf.shape[0]
        self.Nout = self.Neventstot - self.Nin

        newdf["Bg_pdf_loc"] = self.sample.sampleinfo.bgrate.score_samples(
            np.sin(newdf["DECL_EVT"]))

        newdf["Sg_pdf_loc"] = self.sample.sampleinfo.signalPSF(
            newdf["ang_separation"])

        return newdf

    def StdLikelihood(self, params):
        """Function which calculates the Standard Likelihood for a given value
        of ns.

        Parameters
        ----------

        params: list or numpy array
        List of parameters of the likelihood function
        """
        Ntot = self.Neventstot
        ns = float(params[0])
        Sg = self.df_cluster["Sg_pdf_loc"]
        Bg = self.df_cluster["Bg_pdf_loc"]
        insum = np.sum(np.log(ns/Ntot*Sg + (1-ns/Ntot)*Bg*1./(2*math.pi)))
        outsum = self.Nout*np.log(1-ns/Ntot)
        return -(insum + outsum)

    def TScalculation(self, iteration):
        """Function which calculates the Test Statistic for a given cluster.

        Parameters
        ----------

        iteration: int
        Iteration number of the pseudoexperiment
        """

        results = minimize(self.StdLikelihood, self.Nin,
                           bounds=[(0.001, self.Nin)],
                           method="SLSQP")
                           #method="L-BFGS-B")

        LikMax = results.fun
        Lbg = np.sum(np.log(1./(2*math.pi)*self.df_cluster["Bg_pdf_loc"]))
        TS = -LikMax - Lbg

        nsmax = results.x

        if iteration % 100 == 0:
            print("\nITERATION#", iteration)
            print("TS: ", TS)
            print("ns: ", nsmax)
            print("LikMax:", LikMax)
            print("Lbg: ", Lbg)

        opt_values = {"TS": TS, "ns": nsmax, "LikMax": LikMax, "Lbg": Lbg}
        return opt_values
