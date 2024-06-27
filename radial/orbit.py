#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import scipy.optimize as sp

"""
This module computes the radial velocities of a massive object being orbited by
a secondary massive object, and is based on the formalism from Murray & Correia
(2011), available freely at https://arxiv.org/abs/1009.1738. The equation numbers
are from this article, unless otherwise noted.
"""


class BinarySystem(object):
    """
    A class that computes the radial velocities given the orbital parameters of
    the binary system.

    Parameters
    ----------
    k : scalar
        The radial velocity semi-amplitude K in m / s.

    period : scalar
        The orbital period in days.

    t0 : scalar
        Time of pariastron passage in days.

    omega : scalar or ``None``, optional
        Argument of periapse in radians. If ``None``, both ``sqe_cosw`` and
        ``sqe_sinw`` will be required. Default is ``None``.

    ecc : scalar or ``None``, optional
        Eccentricity of the orbit. If ``None``, both ``sqe_cosw`` and
        ``sqe_sinw`` will be required. Default is ``None``.

    sqe_cosw : scalar or ``None``, optional
        The square root of the eccentricity multiplied by the cosine of the
        argument of periapse. If ``None``, both ``omega`` and ``ecc`` will be
        required. Default is ``None``.

    sqe_sinw : scalar or ``None``, optional
        The square root of the eccentricity multiplied by the sine of the
        argument of periapse. If ``None``, both ``omega`` and ``ecc`` will be
        required. Default is ``None``.

    gamma : scalar or ``None``, optional
        Proper motion of the barycenter in m / s. Default is 0.
    """
    def __init__(self, k, period, t0, omega=None, ecc=None, sqe_cosw=None,
                 sqe_sinw=None, gamma=0):

        self.k = k
        self.period = period
        self.t0 = t0
        self.gamma = gamma

        if omega is None or ecc is None:
            if sqe_cosw is None or sqe_sinw is None:
                raise ValueError('Either of these pairs have to be provided: '
                                 '(omega, ecc) or (sqe_cosw, sqe_sinw)')
            else:
                self.ecc = sqe_sinw ** 2 + sqe_cosw ** 2
                self.omega = np.arctan2(sqe_sinw, sqe_cosw)
        else:
            self.ecc = ecc
            self.omega = omega

        if self.ecc > 1:
            raise ValueError('Keplerian orbits are ellipses, therefore ecc <= '
                             '1')

    # Compute Eq. 65
    def rv_eq(self, f):
        """
        The radial velocities equation.

        Parameters
        ----------
        f : scalar or ``numpy.ndarray``
            True anomaly in radians.

        Returns
        -------
        rvs : scalar or ``numpy.ndarray``
            Radial velocity
        """
        rv = self.gamma + self.k * (np.cos(self.omega + f) + self.ecc *
                                    np.cos(self.omega))
        return rv

    # Calculates the Kepler equation (Eq. 41)
    def kep_eq(self, e_ano, m_ano):
        """
        The Kepler equation.

        Parameters
        ----------
        e_ano : scalar
            Eccentric anomaly in radians.

        m_ano : scalar
            Mean anomaly in radians.

        Returns
        -------
        kep: scalar
            Value of E-e*sin(E)-M
        """
        kep = e_ano - self.ecc * np.sin(e_ano) - m_ano
        return kep

    # Calculates the radial velocities for given orbital parameters
    def get_rvs(self, ts):
        """
        Computes the radial velocity given the orbital parameters.

        Parameters
        ----------
        ts : scalar or ``numpy.ndarray``
            Time in days.

        Returns
        -------
        rvs : scalar or ``numpy.ndarray``
            Radial velocities
        """
        m_ano = 2 * np.pi / self.period * (ts - self.t0)  # Mean anomaly
        e_ano = np.array([sp.newton(func=self.kep_eq, x0=mk, args=(mk,))
                          for mk in m_ano])      # Eccentric anomaly
        # Computing the true anomaly
        f = 2 * np.arctan2(np.sqrt(1. + self.ecc) * np.sin(e_ano / 2),
                           np.sqrt(1. - self.ecc) * np.cos(e_ano / 2))
        # Why do we compute the true anomaly in this weird way? Because
        # arc-cosine is degenerate in the interval 0-360 degrees.
        rvs = self.rv_eq(f)
        return rvs
