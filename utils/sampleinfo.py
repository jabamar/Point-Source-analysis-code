"""
Class SampleProperties and some associated functions.
Sample Info contains the relevant distributions needed to understand a given
data sample
"""
import pandas as pd
import numpy as np
import math
# import matplotlib
# import matplotlib.pyplot as plt
from KDEBoundaries1D import KernelDensityBoundaries1D


def rotate_angle(local_coord=[0, 0], source_loc=[np.pi, 0]):
    """
    Calculate the equatorial coordinates of an event given the distance
     to the source, the angle, and the source location.

     Parameters
     ----------

     local_coord: list of 2 floats
     [distance, angle] of the event to the source

     source_loc: list of 2 floats
     [ra, decl] of the source
    """

    dist_local, ang_local = local_coord
    ra_source, decl_source = source_loc

    theta = np.pi/2 - decl_source
    phi = ra_source
    psi = 0

    sindist = math.sin(dist_local)
    cosdist = math.cos(dist_local)
    sinang = math.sin(ang_local)
    cosang = math.cos(ang_local)

    sintheta = math.sin(theta)
    costheta = math.cos(theta)
    sinphi = math.sin(phi)
    cosphi = math.cos(phi)
    sinpsi = math.sin(psi)
    cospsi = math.cos(psi)

    vector_local = [sindist*cosang, sindist*sinang, cosdist]
    vector_final = [vector_local[0]*(-cospsi*sinphi - costheta*cosphi*sinpsi)
                    + vector_local[1]*(sinpsi*sinphi - costheta*cosphi*cospsi)
                    + vector_local[2]*sintheta*cosphi,
                    vector_local[0]*(cospsi*cosphi - costheta*sinphi*sinpsi)
                    + vector_local[1]*(-sinpsi*cosphi - costheta*sinphi*cospsi)
                    + vector_local[2]*sintheta*sinphi,
                    vector_local[0]*sintheta*sinpsi
                    + vector_local[1]*sintheta*cospsi
                    + vector_local[2]*costheta]

    decl_evt = np.pi/2 - math.acos(vector_final[2])
    ra_evt = math.atan2(vector_final[1], vector_final[0])

    return decl_evt, ra_evt


class SampleProperties:
    """Class which contains the relevant properties/pdfs of a sample.

    Attributes
    ----------

    samplename: string
        Name of the sample

    sigmaPSF: float
        sigma value for the Point Spread Function (assuming gaussian dist)

    badrecorate: float
        rate (between 0 and 1) of signal events which we simulate as being
        wrongly reconstructed

    bginfile: string
        Name of the input bg file

    bgrate: KernelDensityBoundaries1D
        Distribution (fitted to a KDE) of background events



    """

    def __init__(self, dictoptions=None):
        """ Class initialization

        Parameters
        ----------

        dictoptions: dictionary
            Dictionary with all the required inputs to initialise the class

        """

        if dictoptions is None:
            raise RuntimeError("Dictionary with SampleProperties info"
                               "not loaded")

        self.samplename = dictoptions["samplename"]
        self.sigmaPSF = dictoptions["sigmaPSF"]
        self.badrecorate = dictoptions["badrecorate"]
        self.bginfile = dictoptions["bginfile"]

        self.read_bgrate()

    def read_bgrate(self):
        """ Takes the input file (bginfile) so to make the background
        sin(decl) distribution
        """

        datasample = pd.read_csv(self.bginfile, delimiter="\t", skiprows=[1])
        datasample["sindec"] = datasample["Decl"].apply(lambda x:
                                                        np.sin(np.deg2rad(x)))
        # histogram = plt.hist(datasample["sindec"], bins=40, range=[-1,1])

        self.NTotalEvents = len(datasample["sindec"])

        bw_silverman = np.std(datasample["sindec"]) * \
            np.power(4/(3*len(datasample["sindec"])), 1./5)

        kderefl = KernelDensityBoundaries1D(kernel="gaussian",
                                            boundary="reflection",
                                            range=[-1, 0.8],
                                            bandwidth=bw_silverman,
                                            n_approx=100)

        kderefl.fit(datasample[["sindec"]])

        self.bgrate = kderefl

    def return_bgrate(self):
        return self.bgrate

    def generate_bgevts(self):
        """ Generate background events by assuming the total number
        of background events follows a Poisson distribution
        """
        NEvtsSample = np.random.poisson(self.NTotalEvents, 1)
        DECL = np.arcsin(self.bgrate.generate_random(size=NEvtsSample))
        RA = np.random.uniform(0, 2*np.pi, size=NEvtsSample)
        return RA, DECL

    def generate_signal(self, Nsignal, signalposdeg=[180, -40]):
        """
        Generate signal events from a gaussian PSF
        First, distance to source is generated for all events. Then, this is
        translated into an event in Equatorial coordinates.
        Returns two lists with the ra, decl of each event IN RADIANS!

        Parameters
        ----------

        Nsignal: int
        Number of signal events to be generated

        signalposdeg: list of two floats
        Location of the source in equatorial coordinates [ra, decl]
        """

        distributedevents = np.random.binomial(1, 1-self.badrecorate, Nsignal)
        Nwellreco = len(distributedevents[distributedevents == 1])
        Nbadlyreco = len(distributedevents[distributedevents == 0])

        distance_to_source = np.abs(
            np.random.normal(scale=np.radians(self.sigmaPSF),
                             size=Nwellreco))
        npoints = len(distance_to_source[distance_to_source > 2*np.pi])

        # Gaussian PSF is an approximation -> We don't want events with
        # angular distances larger than 180 deg!
        while(npoints > 0):
            newpts = np.abs(np.random.normal(scale=np.radians(self.sigma),
                                             size=npoints))
            distance_to_source[distance_to_source > 2*np.pi] = newpts
            npoints = len(distance_to_source[distance_to_source > 2*np.pi])

        badly_reco = np.abs(np.arcsin(np.random.uniform(0, 1, size=Nbadlyreco)))

        # distance_to_source contains all event distances from real source
        distance_to_source = np.concatenate([distance_to_source, badly_reco])

        angle_from_source = np.abs(np.arcsin(np.random.uniform(0, 1,
                                                               size=Nsignal)))

        signalpos = np.radians(signalposdeg)

        ra_signal_evts = []
        decl_signal_evts = []

        for dist, angle in zip(distance_to_source, angle_from_source):
            decl_evt, ra_evt = rotate_angle(local_coord=[dist, angle],
                                            source_loc=signalpos)

            ra_signal_evts.append(ra_evt)
            decl_signal_evts.append(decl_evt)

        return ra_signal_evts, decl_signal_evts
