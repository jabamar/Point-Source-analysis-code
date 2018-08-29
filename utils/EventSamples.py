import pandas as pd
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from .sampleinfo import SampleProperties
import matplotlib.pyplot as plt


class EventSample:
    """Class which contains events of the sample.

    Attributes
    ----------

    sampleinfo: SampleProperties
    SampleProperties element with all the required properties/functions to
    generate the events in the sample.

    df_events: DataFrame
    DataFrame with the events
    """

    def __init__(self, sampleinfo=SampleProperties()):
        self.sampleinfo = sampleinfo
        self.df_events = pd.Dataframe()

    def generate_events(self, Nsignal=0, signalpos=[180, -40]):
        """
        Generates signal and background events and it stores them into a
        DataFrame.

        Parameters
        ----------

        Nsignal: int
        Number of signal events

        signalpos: list of two floats
        [RA, DEC] of the source in degrees
        """

        ra_sg, decl_sg = self.sampleinfo.generate_signal(Nsignal, signalpos)
        ra_bg, decl_bg = self.sampleinfo.generate_bgevts()

        RA_EVTS = np.concatenate([ra_sg, ra_bg])
        DECL_EVTS = np.concatenate([decl_sg, decl_bg])

        self.df_events = pd.DataFrame({"RA_EVT": RA_EVTS,
                                       "DECL_EVT": DECL_EVTS})

    def PlotEventMap(self):
        """
        Makes a plot in equatorial coordinates of the event sample
        """

        # Galactic plane line
        lon_array = np.arange(0, 360)
        lat = np.zeros(360)
        eq_array = np.zeros((360, 2))

        ga = SkyCoord(frame="galactic", l=lon_array, b=lat, unit=u.degree)
        eq_array = ga.icrs

        RA_galplane = eq_array.ra.degree - 180
        Dec_galplane = eq_array.dec.degree

        RA_evts = np.pi - self.df_events["RA_EVT"]
        Dec_evts = self.df_events["DECL_EVT"]

        plt.figure(figsize=(16, 10))

        # Moll projection
        hp.mollview(title="", hold=True)
        hp.graticule(dpar=30, dmer=30)  # 45, 60

        # Labels
        plt.text(2.02, 0, "0h", fontsize=20)
        plt.text(-2.2, 0, "24h", fontsize=20)
        plt.text(-1.6, 0.74, r"+60$^\circ$", fontsize=20)
        plt.text(-2.1, 0.39, r"+30$^\circ$", fontsize=20)
        plt.text(-2.1, -0.43, r"-30$^\circ$", fontsize=20)
        plt.text(-1.6, -0.79, r"-60$^\circ$", fontsize=20)

        hp.projplot(RA_galplane, Dec_galplane, '--', coord='E', lw=2,
                    lonlat=True, color="#000000", markersize=1,
                    markeredgecolor="none")  # 2163B9

        hp.projplot(np.degrees(RA_evts), np.degrees(Dec_evts), 'x',
                    color="#0368ff", coord='E', lonlat=True, markersize=2)

        plt.show()
