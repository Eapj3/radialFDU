#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.table as tb
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd

"""
This module is used to manage radial velocities datasets.
"""


# The RV dataset class
class RVDataSet(object):
    """
    Read and store the data and information of the radial velocities dataset in
    an intelligible manner. This class utilizes the power and convenience of
    ``astropy`` units and tables.

    Parameters
    ----------
    file : ``str``
        Name of the file that contains the radial velocities data.

    t_col : ``int``, optional
        Column number of the data file that corresponds to time. Default is 0.

    rv_col : ``int``, optional
        Column number of the data file that corresponds to radial velocities.
        Default is 1.

    rv_unc_col : ``int``, optional
        Column number of the data file that corresponds to radial velocity
        uncertainties. Default is 2.

    skiprows : ``int``, optional
        Number of rows to skip from the data file. Default is 0.


    t_offset : ``float``, ``astropy.units.Quantity`` or ``None``, optional
        Numerical offset to be summed to the time array. If ``None``, no offset
        is applied. Default is ``None``.

    rv_offset : ``str``, ``float``, ``astropy.units.Quantity`` or ``None``,
                optional
        Numerical offset to be summed to the radial velocities array. If
        ``None``, no offset is applied. ``str`` options are 'subtract_median'
        and 'subtract_mean' (self-explanatory). Default is ``None``.

    t_unit : ``astropy.units`` or ``None``, optional
        The unit of the time array, in ``astropy.units``. If ``None``, uses
        days. Default is ``None``.

    rv_unit : ``astropy.units`` or ``None``, optional
        The unit of the radial velocities and uncertainties arrays, in
        ``astropy.units``. If ``None``, uses km/s. Default is ``None``.

    instrument_name : ``str`` or ``None``, optional
        Name of the instrument, which will be saved in the metadata. Default is
        ``None``.

    target_name : ``str`` or ``None``, optional
        Name of the observed target, which will be saved in the metadata.
        Default is ``None``.

    other_meta : ``dict`` or ``None``, optional
        Other metadata to be saved in the table. If ``None``, no addition is
        made. Default is ``None``.
    """
    def __init__(self, file, t_col=0, rv_col=1, rv_unc_col=2, skiprows=0,
                 t_offset=None, rv_offset=None, t_unit=None,
                 rv_unit=None, instrument_name=None, target_name=None,
                 other_meta=None):

        # TODO: Add option to set the plotting parameters (symbol, color etc.)

        # Setting up the default values
        # The time unit
        if t_unit is None:
            self.t_unit = u.d
        else:
            self.t_unit = t_unit
        # The RV unit
        if rv_unit is None:
            self.rv_unit = u.km / u.s
        else:
            self.rv_unit = rv_unit
        # Other stuff
        if instrument_name is None:
            self.instr_name = 'Unnamed instrument'
        else:
            self.instr_name = instrument_name
        if target_name is None:
            self.target_name = 'Unnamed target'
        else:
            self.target_name = target_name

        # Read the data from file
        data_frame = pd.read_csv(file, skiprows=skiprows)

        self.t = np.array(data_frame.iloc[:, t_col]) * self.t_unit

        self.rv = np.array(data_frame.iloc[:, rv_col]) * self.rv_unit

        self.rv_unc = np.array(data_frame.iloc[:, rv_unc_col]) * self.rv_unit

        # The offsets
        if t_offset is None:
            self.t_offset = 0
        elif isinstance(t_offset, u.Quantity) is True:
            self.t_offset = t_offset
        else:
            self.t_offset = t_offset * self.t_unit
        # The RV offset
        if rv_offset is None:
            self.rv_offset = 0
        elif isinstance(rv_offset, u.Quantity) is True:
            self.rv_offset = rv_offset
        elif rv_offset == 'subtract_median':
            self.rv_offset = -np.median(self.rv)
        elif rv_offset == 'subtract_mean':
            self.rv_offset = -np.mean(self.rv)
        else:
            self.rv_offset = rv_offset * self.rv_unit

        self.t += self.t_offset
        self.rv += self.rv_offset

        # Create an astropy table with the data
        self.table = tb.Table([self.t, self.rv, self.rv_unc],
                              names=['Time', 'RV', 'RV sigma'],
                              meta={'Instrument': self.instr_name,
                                    'Time offset': self.t_offset,
                                    'RV offset': self.rv_offset,
                                    'Target': self.target_name})

        # Add the optional extra metadata
        if isinstance(other_meta, dict):
            self.table.meta.update(other_meta)

    # Plot the data
    def plot(self):
        """
        Plot the data set.
        """
        plt.errorbar(self.t.value, self.rv.value, yerr=self.rv_unc.value,
                     fmt='ko')
        plt.title("{} observed on {}".format(self.table.meta['Target'],
                                             self.table.meta['Instrument']))
        plt.xlabel('Time ({})'.format(self.t.unit))
        plt.ylabel('Radial velocities ({})'.format(self.rv.unit))
        plt.show()
