#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
from radial import orbit, dataset, rv_model, prior
import scipy.signal as ss
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
import matplotlib.gridspec as gridspec
import lmfit
import corner
import emcee
import astropy.units as u
import astropy.constants as c

"""
This code contains routines to estimate the orbital parameters of a binary
system by means of maximum likelihood estimation or a Markov-Chain Monte Carlo
estimation using the code emcee. Before using it, it is highly recommended to
read the documentation of emcee in order to understand what are priors,
probabilities, sampling and other jargon. Check it out at
http://dan.iel.fm/emcee/current/
"""


# Estimate orbital parameters from radial velocity data comprising at least one
# orbit.
class FullOrbit(object):
    """
    A class that computes the orbital parameters of a binary system given its
    radial velocities (and their uncertainties) in function of time. This class
    is optimized for time series that contain at least one full or almost full
    orbital period.

    Parameters
    ----------
    datasets : sequence or ``radial.dataset.RVDataSet``
        A list of ``RVDataSet`` objects or one ``RVDataSet`` object that
        contains the data to be fit. If a sequence is passed, the order that
        the data sets in the sequence will dictate which instrumental parameter
        (gamma, sigma) index correspond to each data set.

    guess : ``dict``
        First guess of the orbital parameters. The keywords must match to the
        names of the parameters to be fit. These names are: ``'k'``,
        ``'period'``, ``'t0'``, ``'omega'``, ``'ecc'``, ``'gamma_X'``,
        ``'sigma_X'`` (and so forth), where 'X' is the index of the data set.

    use_add_sigma : ``bool``, optional
        If ``True``, the code will use additional parameter to estimate an extra
        uncertainty term for each RV data set. Default is ``False``.

    parametrization : ``str``, optional
        The parametrization for the orbital parameter search. Currently
        available options are ``'mc10'`` and ``'exofast'``. Default is
        ``'mc10'``.
    """
    def __init__(self, datasets, guess, use_add_sigma=False,
                 parametrization='mc10'):

        self.datasets = datasets
        self.guess = guess
        if parametrization == 'mc10' or parametrization == 'exofast':
            self.parametrization = parametrization
        else:
            raise ValueError('parametrization must be "mc10" or "exofast".')

        if isinstance(datasets, dataset.RVDataSet):
            self.n_ds = 1
        else:
            self.n_ds = len(datasets)
            # Check if the datasets are passed as RVDataSet objects
            for dsk in self.datasets:
                assert isinstance(dsk,
                                  dataset.RVDataSet), 'The datasets must be ' \
                                                      'passed as RVDataSet ' \
                                                      'objects.'

        # Read the data
        self.t = []
        self.rv = []
        self.rv_unc = []
        self.meta = []
        for dsk in self.datasets:
            self.t.append(dsk.t.to(u.d).value)
            self.rv.append(dsk.rv.to(u.m / u.s).value)
            self.rv_unc.append(dsk.rv_unc.to(u.m / u.s).value)
            self.meta.append(dsk.   table.meta)

        self.use_add_sigma = use_add_sigma

        # The global parameter keywords to be used in the code
        self.keys = list(self.guess.keys())

        # Initializing useful global variables
        self.lmfit_result = None
        self.lmfit_chisq = None
        self.residuals = None
        self.sampler = None
        self.emcee_chains = None
        self.ndim = None
        self.best_params = {}
        for key in self.keys:
            self.best_params[key] = None

    # Compute a periodogram of a data set
    def lomb_scargle(self, dset_index, freqs):
        """
        Compute a Lomb-Scargle periodogram for a given data set using
        ``scipy.signal.lombscargle``.

        Parameters
        ----------
        dset_index : ``int``
            Index of the data set to have the periodogram calculated for.

        freqs : ``array_like``
            Angular frequencies for output periodogram.

        Returns
        -------
        pgram : ``array_like``
            Lomb-Scargle periodogram.

        fig : ``matplotlib.pyplot.figure``
            Periodogram plot.
        """
        x_array = self.t[dset_index]
        y_array = self.rv[dset_index]
        pgram = ss.lombscargle(x=x_array, y=y_array, freqs=freqs)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.semilogx(freqs, pgram)
        return pgram, fig

    # Plot the data sets
    def plot_rvs(self, legend_loc=None, symbols=None, plot_guess=False,
                plot_samples=False, fold=False, numpoints=1000):
        """
        Plot the data sets.

        Parameters
        ----------
        legend_loc : ``int`` or ``None``, optional
            Location of the legend. If ``None``, use the default from
            ``matplotlib``. Default is ``None``.

        symbols : sequence or ``None``, optional
            List of symbols for each data set in the plot. If ``None``, use
            the default list from ``matplotlib`` markers. Default is ``None``.

        plot_guess : ``bool``, optional
            If ``True``, also plots the guess as a black curve, and the RVs of
            each data set is shifted by its respective gamma value.

        plot_samples : ``bool``, optional
            If ``True``, also plots the samples obtained with the ``emcee``
            estimation. Default is ``False``.

        fold : ``bool``, optional
            If ``True``, plot the radial velocities by folding them around the
            estimated orbital period. Default is ``False``.

        numpoints : ``int``, optional
            Number of points to compute the radial velocities curve. Default is
            ``1000``.
        """
        # Plot of emcee samples is not implemented yet.
        if plot_samples is True:
            raise NotImplementedError('Plot of emcee samples is not supported'
                                      'yet.')


        # Use matplotlib's default symbols if ``None`` is passed.
        if symbols is None:
            markers = mrk.MarkerStyle("")
            symbols = markers.filled_markers

        fig = plt.figure(figsize=(6, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=(4, 1))
        ax_fit = fig.add_subplot(gs[0])

        self.residuals = []
        for i in range(self.n_ds):
            if plot_guess is True:
                ax_res = fig.add_subplot(gs[1], sharex=ax_fit)

                # First we figure out the bounds of the plot
                t_min = min([min(tk) for tk in self.t])
                t_max = max([max(tk) for tk in self.t])
                if fold is False:
                    t_guess = np.linspace(t_min, t_max, numpoints)
                else:
                    t_guess = np.linspace(0, 1, numpoints)

                # Compute the radial velocities for the guess
                system = orbit.BinarySystem(k=self.guess['k'],
                                            period=self.guess['period'],
                                            t0=self.guess['t0'],
                                            omega=self.guess['omega'],
                                            ecc=self.guess['ecc'],
                                            gamma=0)

                if fold is False:
                    rv_guess = system.get_rvs(ts=t_guess)
                else:
                    rv_guess = system.get_rvs(ts=t_guess * self.guess['period'])
                rv_guess_samepoints = system.get_rvs(ts=self.t[i])

                # Shift the radial velocities with the provided gamma
                rvs = self.rv[i] - self.guess['gamma_{}'.format(i)]

                # Compute residuals
                res = rv_guess_samepoints - rvs
                self.residuals.append(res)

                # And finally
                if fold is False:
                    # Plot the data and the curve
                    ax_fit.errorbar(self.t[i], rvs, yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])
                    ax_fit.plot(t_guess, rv_guess, color='k')
                    # Plot the residuals
                    ax_res.errorbar(self.t[i], res, yerr=self.rv_unc[i],
                                    fmt=symbols[i])
                    ax_res.set_ylabel('Residuals\n(m s$^{-1}$)')
                    # Trick to make y-axis in residuals symmetric
                    y_min = abs(np.min(res - self.rv_unc[i]))
                    y_max = abs(np.max(res + self.rv_unc[i]))
                    y_limit = max([y_min, y_max]) * 1.1
                    ax_res.set_ylim(-y_limit, y_limit)
                    ax_res.axhline(y=0.0, linewidth=1, color='k', ls='--')
                    plt.setp(ax_res.get_xticklabels(), visible=False)
                else:
                    # Plot the data and the curve
                    phase = (self.t[i] / self.guess['period']) % 1
                    ax_fit.errorbar(phase, rvs, yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])
                    ax_fit.plot(t_guess, rv_guess, color='k')
                    ax_fit.set_xlim(0.0, 1.0)
                    # Plot the residuals
                    ax_res.errorbar(phase, res, yerr=self.rv_unc[i],
                                    fmt=symbols[i])
                    ax_res.set_ylabel('Residuals\n(m s$^{-1}$)')
                    # Trick to make y-axis in residuals symmetric
                    y_min = abs(np.min(res - self.rv_unc[i]))
                    y_max = abs(np.max(res + self.rv_unc[i]))
                    y_limit = max([y_min, y_max]) * 1.1
                    ax_res.set_ylim(-y_limit, y_limit)
                    ax_res.axhline(y=0.0, linewidth=1, color='k', ls='--')
                    plt.setp(ax_res.get_xticklabels(), visible=False)
            else:
                if fold is False:
                    ax_fit.errorbar(self.t[i], self.rv[i], yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])
                else:
                    phase = (self.t[i] / self.guess['period']) % 1
                    ax_fit.errorbar(phase, self.rv[i], yerr=self.rv_unc[i],
                                    fmt=symbols[i],
                                    label=self.meta[i]['Instrument'])

        # Show the plot
        if fold is False:
            ax_fit.set_xlabel('Time (d)')
        else:
            ax_fit.set_xlabel('Phase')
        ax_fit.set_ylabel('Radial velocities (m s$^{-1}$)')
        ax_fit.set_title('{}'.format(self.meta[0]['Target']))
        ax_fit.legend(loc=legend_loc, numpoints=1)
        plt.tight_layout()

        if plot_guess is False:
            return ax_fit
        else:
            return fig, gs

    # The log-likelihood
    def lnlike(self, theta):
        """
        Log-likelihood of a given set of parameters to adequately describe the
        observed data.

        Parameters
        ----------
        theta : ``dict`` or ``lmfit.Parameters``
            The orbital parameters.

        Returns
        -------
        sum_res : scalar
            The log-likelihood.
        """
        try:
            # If ``theta`` is an ``lmfit.Parameters`` object
            v = theta.valuesdict()
        except AttributeError:
            # If ``theta`` is a regular ``dict`` object
            v = theta

        sum_res = 0

        for i in range(self.n_ds):

            # Compute the RVs using the appropriate model
            if self.parametrization == 'mc10':
                rvs = rv_model.mc10(self.t[i], v['log_k'], v['log_period'],
                                    v['t0'], v['omega'], v['ecc'],
                                    v['gamma_{}'.format(i)])
            elif self.parametrization == 'exofast':
                rvs = rv_model.exofast(self.t[i], v['log_k'], v['log_period'],
                                    v['t0'], v['sqe_cosw'], v['sqe_sinw'],
                                    v['gamma_{}'.format(i)])

            # If user wants to estimate additional sigma
            if self.use_add_sigma is False:
                inv_sigma2 = (1. / (self.rv_unc[i] ** 2))
            elif self.use_add_sigma is True:
                log_sigma_j = theta['log_sigma_{}'.format(i)]
                inv_sigma2 = (1. / (self.rv_unc[i] ** 2 +
                                    (10 ** log_sigma_j) ** 2))

            # The log-likelihood
            sum_res += np.sum((self.rv[i] - rvs) ** 2 * inv_sigma2 +
                              np.log(2. * np.pi / inv_sigma2))
        return sum_res

    # Prepare an ``lmfit.Parameters`` object
    def prepare_params(self, theta, vary_param=None):
        """
        Prepare a ``lmfit.Parameters`` object to be used in the ``lmfit``
        estimation.

        Parameters
        ----------
        theta : ``dict``
            The current orbital parameters.

        vary_param : ``dict``
            Dictionary that says which parameters will vary in the estimation.
            By default, all parameters vary. A parameter can be fixed by setting
            its key to ``False``.

        Returns
        -------
        params : ``lmfit.Parameters``
            The ``lmfit.Parameters`` object for the estimation.
        """
        # Setting up the vary params dict
        vary = {}
        keys = list(theta.keys())
        for key in keys:
            vary[key] = True

        if vary_param is not None:
            for key in keys:
                try:
                    vary[key] = vary_param[key]
                except KeyError:
                    pass

        params = lmfit.Parameters()

        # Set the bounds for omega and ecc or sqe_cosw and sqe_sinw
        if self.parametrization == 'mc10':
            bounds = {'omega': (-np.pi, np.pi), 'ecc': (1E-6, 0.99999)}
        elif self.parametrization == 'exofast':
            bounds = {'sqe_cosw': (-1, 1), 'sqe_sinw': (-1, 1)}

        # Generate the parameters object
        for key in keys:
            try:
                params.add(key, theta[key], vary=vary[key],
                           min=bounds[key][0], max=bounds[key][1])
            except KeyError:
                params.add(key, theta[key], vary=vary[key])
        return params

    # Estimation using lmfit
    def lmfit_orbit(self, vary=None, verbose=True, update_guess=False,
                    minimize_mode='Nelder'):
        """
        Perform a fit to the radial velocities datasets using
        ``lmfit.minimize``.

        Parameters
        ----------
        vary : ``dict`` or ``None``, optional
            Dictionary with keywords corresponding to each parameter, and
            entries that are ``True`` if the parameter is to be left to vary, or
            ``False`` if the parameter is to be fixed in the value provided by
            the guess. If ``None``, all parameters will vary. Default is
            ``None``.

        verbose : ``bool``, optional
            If ``True``, print output in the screen. Default is ``False``.

        update_guess : ``bool``, optional
            If ``True``, updates ``~estimate.FullOrbit.guess`` with the
            estimated values from the minimization. Default is ``False``.

        minimize_mode : ``str``, optional
            The minimization algorithm string. See the documentation of
            ``lmfit.minimize`` for a list of options available. Default is
            ``'Nelder'``.

        Returns
        -------
        result : ``lmfit.MinimizerResult``
            The resulting ``MinimizerResult`` object.
        """
        # Prepare the ``lmfit.Parameters`` object
        guess = {'log_k': np.log10(self.guess['k']),
                 'log_period': np.log10(self.guess['period']),
                 't0': self.guess['t0']}
        if self.parametrization == 'mc10':
            guess['omega'] = self.guess['omega']
            guess['ecc'] = self.guess['ecc']
        elif self.parametrization == 'exofast':
            guess['sqe_cosw'] = np.sqrt(self.guess['ecc']) * \
                                np.cos(self.guess['omega'])
            guess['sqe_sinw'] = np.sqrt(self.guess['ecc']) * \
                                np.sin(self.guess['omega'])
        for i in range(self.n_ds):
            guess['gamma_{}'.format(i)] = self.guess['gamma_{}'.format(i)]
            if self.use_add_sigma is True:
                guess['log_sigma_{}'.format(i)] = \
                    np.log10(self.guess['sigma_{}'.format(i)])
        params = self.prepare_params(guess, vary)

        # Perform minimization
        self.lmfit_result = lmfit.minimize(self.lnlike, params,
                                           method=minimize_mode)
        self.lmfit_chisq = self.lmfit_result.chisqr

        # Updating global variable best_params
        keys = list(guess.keys())
        for key in keys:
            self.best_params[key] = self.lmfit_result.params[key].value

        if update_guess is True:
            self.guess['k'] = 10 ** self.best_params['log_k']
            self.guess['period'] = 10 ** self.best_params['log_period']
            self.guess['t0'] = self.best_params['t0']
            if self.parametrization == 'mc10':
                self.guess['omega'] = self.best_params['omega']
                self.guess['ecc'] = self.best_params['ecc']
            elif self.parametrization == 'exofast':
                self.guess['omega'] = np.arctan2(self.best_params['sqe_sinw'],
                                                 self.best_params['sqe_cosw'])
                self.guess['ecc'] = self.best_params['sqe_sinw'] ** 2 + \
                                    self.best_params['sqe_cosw'] ** 2
            for i in range(self.n_ds):
                self.guess['gamma_{}'.format(i)] = \
                    self.best_params['gamma_{}'.format(i)]
                if self.use_add_sigma is True:
                    self.guess['sigma_{}'.format(i)] = \
                        10 ** self.best_params['log_sigma_{}'.format(i)]

        if verbose is True:
            for key in keys:
                print('{} = {}'.format(key, self.best_params[key]))

        return self.best_params

    def compute_dynamics(self, main_body_mass=1.0):
        """
        Compute the mass and semi-major axis of the companion defined by the
        orbital parameters.

        Parameters
        ----------
        main_body_mass : ``float``, optional
            The mass of the main body which the companion orbits, in units of
            solar masses. Default is 1.0.
        """
        mbm = main_body_mass * u.solMass
        k = self.guess['k'] * u.m / u.s
        period = self.guess['period'] * u.d
        ecc = self.guess['ecc']

        # Compute mass function f
        f = (period * k ** 3 * (1 - ecc ** 2) ** (3 / 2) /
             (2 * np.pi * c.G)).to(u.solMass)

        # Compute msini
        msini = abs(np.roots([1, -f.value, -2 * mbm.value * f.value,
                              -mbm.value ** 2 * f.value])[0]) * u.solMass

        # Compute the semi-major axis in km and convert to AU
        semi_a = (np.sqrt(c.G * msini * period / k / (2 * np.pi) /
                         np.sqrt(1. - ecc ** 2))).to(u.AU)
        return msini.value, semi_a.value

    # The probability
    def lnprob(self, theta_list):
        """
        This function calculates the ln of the probabilities to be used in the
        MCMC estimation.

        Parameters
        ----------
        theta_list: sequence

        Returns
        -------
        The probability of the signal rv being the result of a model with the
        parameters theta
        """
        # The common parameters
        theta = {'log_k': theta_list[0],
                 'log_period': theta_list[1],
                 't0': theta_list[2]}

        # Parametrization option-specific parameters
        if self.parametrization == 'mc10':
            theta['omega'] = theta_list[3]
            theta['ecc'] = theta_list[4]
        elif self.parametrization == 'exofast':
            theta['sqe_cosw'] = theta_list[3]
            theta['sqe_sinw'] = theta_list[4]

        # Instrumental parameters
        for i in range(self.n_ds):
            theta['gamma_{}'.format(i)] = theta_list[5 + i]
            if self.use_add_sigma is True:
                theta['log_sigma_{}'.format(i)] = theta_list[5 + self.n_ds + i]

        lp = prior.flat(theta, self.parametrization)
        params = self.prepare_params(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - 0.5 * self.lnlike(params)

    # Using emcee to estimate the orbital parameters
    def emcee_orbit(self, nwalkers=20, nsteps=1000, p_scale=2.0, nthreads=1,
                    ballsizes=None):
        """
        Calculates samples of parameters that best fit the signal rv.

        Parameters
        ----------
        nwalkers : ``int``
            Number of walkers

        nsteps : ``int``
            Number of burning-in steps

        p_scale : ``float``, optional
            The proposal scale parameter. Default is 2.0.

        nthreads : ``int``
            Number of threads in your machine

        ballsizes : ``dict``
            The one-dimensional size of the volume from which to generate a
            first position to start the chain.

        Returns
        -------
        sampler : ``emcee.EnsembleSampler``
            The resulting sampler object.
        """

        if ballsizes is None:
            ballsizes = {'log_k': 1E-4, 'log_period': 1E-4, 't0': 1E-4,
                         'omega': 1E-4, 'ecc': 1E-4, 'sqe_cosw': 1E-4,
                         'sqe_sinw': 1E-4, 'gamma': 1E-4, 'log_sigma': 1E-4}

        # The labels
        if self.parametrization == 'mc10':
            self.labels = [r'\log{K}', r'\log{T}', 't_0', r'\omega',
                           'e']
        elif self.parametrization == 'exofast':
            self.labels = [r'\log{K}', r'\log{T}', 't_0',
                           r'\sqrt{e} \cos{\omega}',
                           r'\sqrt{e} \sin{\omega}']
        for i in range(self.n_ds):
            self.labels.append(r'\gamma_{}'.format(i))
            if self.use_add_sigma is True:
                self.labels.append(r'\log{\sigma_%s}' % str(i))

        # Creating the pos array
        pos = []
        for n in range(nwalkers):
            pos_n = [np.log10(self.guess['k']) + ballsizes['log_k'] *
                     np.random.normal(),
                     np.log10(self.guess['period']) + ballsizes['log_period'] *
                     np.random.normal(),
                     self.guess['t0'] + ballsizes['t0'] * np.random.normal()]
            if self.parametrization == 'mc10':
                pos_n.append(self.guess['omega'] + ballsizes['omega'] *
                             np.random.normal())
                pos_n.append(self.guess['ecc'] + ballsizes['ecc'] *
                                        np.random.normal())
            elif self.parametrization == 'exofast':
                sqe_cosw = np.sqrt(self.guess['ecc']) * \
                           np.cos(self.guess['omega'])
                sqe_sinw = np.sqrt(self.guess['ecc']) * \
                           np.sin(self.guess['omega'])
                pos_n.append(sqe_cosw + ballsizes['sqe_cosw'] *
                             np.random.normal())
                pos_n.append(sqe_sinw + ballsizes['sqe_sinw'] *
                             np.random.normal())
            for i in range(self.n_ds):
                pos_n.append(self.guess['gamma_{}'.format(i)] +
                             ballsizes['gamma'] * np.random.normal())
                if self.use_add_sigma is True:
                    pos_n.append(np.log10(self.guess['sigma_{}'.format(i)]) +
                                 ballsizes['log_sigma'] * np.random.normal())
            pos.append(np.array(pos_n))

        self.ndim = len(pos[0])

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob,
                                        a=p_scale, threads=nthreads)
        sampler.run_mcmc(pos, nsteps)
        self.sampler = sampler
        return sampler

    # Plot emcee chains
    def plot_emcee_sampler(self, outfile=None, n_cols=2, fig_size=(12, 12)):
        """
        Plot the ``emcee`` sampler so that the user can check for convergence.

        Parameters
        ----------
        outfile : ``str`` or ``None``, optional
            Name of the output image file to be saved. If ``None``, then no
            output file is produced, and the plot is displayed on screen.
            Default is ``None``.

        n_cols : ``int``, optional
            Number of columns of the plot. Default is 2.

        fig_size : tuple, optional
            Sizes of each panel of the plot, where the first element of the
            tuple corresponds to the x-direction size, and the second element
            corresponds to the y-direction size. Default is (12, 12).
        """
        # TODO: include option to plot the guess in the chains

        assert (self.sampler is not None), "The emcee sampler must be run " \
                                           "before plotting the chains."
        n_walkers, n_steps, n_params = np.shape(self.sampler.chain)

        # Dealing with the number of rows for the plot
        if n_params % n_cols > 0:
            n_rows = n_params // n_cols + 1
        else:
            n_rows = n_params // n_cols

        # Finally Do the actual plot
        ind = 0  # The parameter index
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, sharex=True,
                                 figsize=fig_size)
        for i in range(n_cols):
            for k in range(n_rows):
                if ind < len(self.labels):
                    axes[k, i].plot(self.sampler.chain[:, :, ind].T)
                    axes[k, i].set_ylabel(r'$%s$' % (self.labels[ind]))
                    ind += 1
                else:
                    pass
            plt.xlabel('Step number')
        if outfile is None:
            plt.show()
        else:
            plt.savefig(outfile)

    # Make chains from sampler
    def make_chains(self, ncut, outfile=None):
        """
        Make a chains object that represent the posterior distributions of the
        orbital parameters.

        Parameters
        ----------
        ncut : ``int``
            Number of points of the burn-in phase to be ignored.

        outfile : ``str`` or ``None``
            A string containing the path to the file where the chains will be
            saved. This is useful when you do not want to keep running ``emcee``
            frequently. If ``None``, no output file is produced. Default is
            ``None``.

        Returns
        -------
        emcee_chains : ``numpy.ndarray``
            The chains of the ``emcee`` run, with the burn-in phase removed.
        """
        emcee_chains = self.sampler.chain[:, ncut:, :].reshape((-1, self.ndim))
        self.emcee_chains = emcee_chains

        # Save chains to file
        if isinstance(outfile, str):
            np.save(outfile, self.emcee_chains)
        elif outfile is None:
            pass
        else:
            raise TypeError('``outfile`` must be a string or None.')

        return emcee_chains

    # Make a corner plot
    def plot_corner(self):
        """
        Produce a corner (a.k.a. triangle) plot of the posterior distributions
        of the orbital parameters estimated with ``emcee``.

        Returns
        -------
        fig :
        """
        fig = corner.corner(self.emcee_chains, labels=self.labels)
        return fig

    # Print emcee result
    def print_emcee_result(self, main_star_mass=None, mass_sigma=None):
        """

        Returns
        -------

        """
        percentiles = zip(*np.percentile(self.emcee_chains, [16, 50, 84],
                                         axis=0))
        result = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), percentiles))

        # Print raw results in a LaTeX friendly manner
        print('Raw results:')
        for i in range(len(self.labels)):
            print(self.labels[i], '= %.5f^{+%.5f}_{-%.5f}' %
                  (result[i][0], result[i][1], result[i][2]))

        # Work out the human-friendly results
        hf_labels = ['K', 'T', 't_0', r'\omega', 'e']
        units = ['m / s', 'd', 'd', 'deg', ' ']
        hf_chains = np.array(self.emcee_chains)         # K and T from log to
        hf_chains[:, 0:2] = 10 ** (hf_chains[:, 0:2])   # linear
        if self.parametrization == 'mc10':
            hf_chains[:, 3] = hf_chains[:, 3] * 180 / np.pi     # rad to degrees
            hf_chains[:, 4] = hf_chains[:, 4]
        elif self.parametrization == 'exofast':
            # Transform sqe_cosw and sqe_sinw to omega and ecc
            omega = (np.arctan2(hf_chains[:, 4], hf_chains[:, 3])) * 180 / np.pi
            ecc = hf_chains[:, 3] ** 2 + hf_chains[:, 4] ** 2
            hf_chains[:, 3] = omega
            hf_chains[:, 4] = ecc
        hf_perc = zip(*np.percentile(hf_chains, [16, 50, 84], axis=0))
        hf_result = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                             hf_perc))

        # Compute mass and semi-major axis of the orbit
        if main_star_mass is not None:
            if mass_sigma is None:
                msm = np.array([main_star_mass for i in
                                range(len(hf_chains[:, 0]))]) * u.solMass
            else:
                msm = np.random.normal(loc=main_star_mass,
                                       scale=mass_sigma,
                                       size=len(hf_chains[:, 0])) * u.solMass
            k = hf_chains[:, 0] * u.m / u.s
            period = hf_chains[:, 1] * u.d
            ecc = hf_chains[:, 4]
            f = (period * k ** 3 * (1 - ecc ** 2) ** (3 / 2) /
                 (2 * np.pi * c.G)).to(u.solMass)
            msini = []
            for i in range(len(f)):
                msini.append(abs(np.roots([1,  -f[i].value,
                                           -2 * msm[i].value * f[i].value,
                                           -msm[i].value ** 2 * f[i].value])[0])
                             )
            msini = np.array(msini) * u.solMass
            semi_a = (np.sqrt(c.G * msini * period / k / (2 * np.pi) /
                              np.sqrt(1. - ecc ** 2))).to(u.AU)
            hf_perc = zip(*np.percentile(np.array([msini.value,
                                                   semi_a.value]).T,
                                         [16, 50, 84], axis=0))
            hf_m_a = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                 hf_perc))

        print('\nHuman-friendly results:')
        for i in range(len(hf_labels)):
            print(hf_labels[i], '= %.5f^{+%.5f}_{-%.5f} %s' %
                  (hf_result[i][0], hf_result[i][1], hf_result[i][2], units[i]))

        try:
            print(r'm \sin{i}= %.5f^{+%.5f}_{-%.5f} solMass' %
                  (hf_m_a[0][0], hf_m_a[0][1], hf_m_a[0][2]))
            print('a= %.5f^{+%.5f}_{-%.5f} AU' %
                  (hf_m_a[1][0], hf_m_a[1][1], hf_m_a[1][2]))
        except NameError:
            pass
