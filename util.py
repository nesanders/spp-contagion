import numpy as np
import pandas as pd
from pybasicbayes.util.text import progprint_xrange
from types import SimpleNamespace
import sklearn
from copy import deepcopy
import joblib
import logging

import pyhawkes
from pyhawkes.models import \
    DiscreteTimeNetworkHawkesModelGammaMixture, \
    DiscreteTimeStandardHawkesModel, \
    ContinuousTimeNetworkHawkesModel

import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
ls_cycle = lambda i: ['-', '--', '-.'][np.mod(i, 3)]

##########################
## Inference functions
##########################

def identify_basis_edge(test_model, p=0.95):
    """Compute the basis edge; the p'th percentile position of the largest basis 
    vector"""
    cume_basis = np.cumsum(test_model.basis.create_basis(), axis=0)
    cume_basis_norm = cume_basis / cume_basis.max(axis=0)
    return np.argmin(np.abs(cume_basis_norm - p), axis=0).max()

def do_pyhawkes_sampling(test_model, N_samples, thinning=1):
    """Run pyhawkes Gibbs sampling on test_model for N_samples."""
    samples = []
    lps = []
    for i in progprint_xrange(N_samples):
        lps.append(test_model.log_probability())
        if np.mod(i, thinning)==0: samples.append(test_model.copy_sample())
        test_model.resample_model()
    return samples, lps, test_model

def do_pyhawkes_sim(config, N_samples=2000, use_samples=1000, thinning=1):
    fitted_model = {
        'N_samples':int(N_samples/thinning), 
        'use_samples': int(use_samples/thinning), 
        'thinning': int(thinning)}
    
    ## Gather data
    timeseries, timeseries_labels, time_shifts = \
        config['timeseries'], config['timeseries_labels'], config['time_shifts']
    ## Instantiate model
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(
        K=timeseries.shape[1], **config['hawkes_params'])
    
    ## Apply time shifts
    assert min(time_shifts) == 0
    timeseries_shift = np.zeros(timeseries.shape, dtype=timeseries.dtype)
    for i in range(len(time_shifts)):
        timeseries_shift[time_shifts[i]:, i] += timeseries[:len(timeseries)-time_shifts[i],i]
    timeseries_shift = np.array(timeseries_shift)
    
    test_model.add_data(timeseries_shift)
    
    ## Archive timeseries data
    fitted_model['timeseries'] = timeseries
    fitted_model['timeseries_labels'] = timeseries_labels
    fitted_model['time_shifts'] = time_shifts

    ## Sample
    fitted_model['samples'], fitted_model['lps'], fitted_model['last_model'] = \
        do_pyhawkes_sampling(test_model, N_samples, thinning=thinning)

    ## Compute the basis edge
    fitted_model['max_basis_days'] = identify_basis_edge(test_model)
    
    return fitted_model

##########################
## Plotting functions
##########################

def plot_prob_convergence(df, model_name, c, output_dir):
    plt.figure()
    plt.plot(np.arange(c.N_samples), c.lps, 'k')
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.axvline(c.N_samples - c.use_samples, ls='dashed', color='orange')
    plt.show()
    plt.savefig(output_dir + 'test_pyhawkes_' + model_name + '_converge.pdf')

def plot_zoom_rate(df, model_name, c, output_dir, plot_start=None, plot_delta=60):
    if plot_start is None:
        ## Pick the event that generated the maximum amount of coverage
        plot_start = df['MP_stories_total'].argmax() - 3
    c.last_model.plot(T_slice=(plot_start-plot_delta, plot_start+plot_delta))
    plt.savefig(output_dir + 'test_pyhawkes_' + model_name + '_plot_zoom_rate.pdf')

def plot_adjacency(df, model_name, c, output_dir):
    plt.figure()
    c.last_model.plot_adjacency_matrix()
    plt.savefig(output_dir + 'test_pyhawkes_' + model_name + '_plot_adjacency.pdf')

def plot_kernel_basis(df, model_name, c, output_dir):
    plt.figure()
    plt.plot(c.last_model.basis.create_basis())
    plt.axvline(c.max_basis_days, ls='dashed', label='Inferred trunctation position')
    plt.legend()
    plt.savefig(output_dir + 'test_pyhawkes_' + model_name + '_plot_basis.pdf')

def plot_weighted_impulse(df, model_name, c, output_dir):
    ## Calculate weighted impulse response after the warmup samples
    imps = np.percentile(
        np.array([s.impulses * s.W for s in c.samples[-c.use_samples:]]), 
        [5,50,95], axis=0)
    fig, axs = plt.subplots(imps.shape[-1], sharex='all')
    if len(kernels) == 1: axs = [axs]
    for i in range(imps.shape[-1]):
        ax = axs[i]
        for j in range(imps.shape[-1]):
            ax.plot(np.arange(len(imps[0,:,j,i])), imps[1,:,j,i], 
                        label=c.timeseries_labels[j], color=colors[j], zorder=1, lw=2)
            ax.fill_between(np.arange(len(imps[0,:,j,i])), imps[0,:,j,i], imps[2,:,j,i], 
                                label=None, alpha=0.5, color=colors[j], zorder=0)
        ax.set_title('Weighted Effect on '+c.timeseries_labels[i])
        ax.set_ylabel('Impulse')
    axs[0].legend(title='Effect of...')
    ax.set_xlabel('Days')
    plt.xlim(0, c.max_basis_days)
    plt.savefig(output_dir + 'test_pyhawkes_' + model_name + '_plot_impulses_90per_weight.pdf')

def plot_trace(df, model_name, c, output_dir):
    fig, axs = plt.subplots(2, 2)
    samp_W = np.array([s.W for s in c.samples])
    px = np.arange(c.N_samples)[-c.use_samples:]
    for k in range(c.timeseries.shape[1]):
        for jk in range(c.timeseries.shape[1]):
            axs[0, 0].plot(samp_W[:, k, jk], 
                    color=colors[k], label=f'K={k}; j={jk}', ls=ls_cycle(jk)) 
            axs[0, 1].plot(px, 
                (samp_W[-c.use_samples:, k, jk] - samp_W[-c.use_samples:, k, jk].mean(axis=0))/samp_W[-c.use_samples:, k, jk].std(axis=0), 
                color=colors[k], ls=ls_cycle(jk)) 
    axs[0, 0].set_ylabel('W; weight adjacency matrix')
    axs[0, 0].legend(fontsize=6)
    samp_lambda0 = np.array([s.lambda0 for s in c.samples])
    axs[1, 0].plot(samp_lambda0)
    axs[1, 1].plot(px, 
        (samp_lambda0[-c.use_samples:] - samp_lambda0[-c.use_samples:].mean(axis=0))/samp_lambda0[-c.use_samples:].std(axis=0)) 
    axs[1, 0].set_ylabel('lambda0; constant background level')
    axs[0, 0].set_title('Raw')
    axs[0, 1].set_title('Normalized')
    axs[-1, 0].set_xlabel('Iteration')
    axs[-1, 1].set_xlabel('Iteration (after trimming)')
    plt.savefig(output_dir + 'test_pyhawkes_' + model_name + '_traces.pdf')

def do_pyhawkes_plots(df, model_name, fitted_model, output_dir=''):
    ## Load the parameters of the fitted model
    c = SimpleNamespace(**fitted_model)
    
    ## Log prob convergence
    plot_prob_convergence(df, model_name, c, output_dir)
    
    ## Plot rate chart zoomed in near an event
    plot_zoom_rate(df, model_name, c, output_dir)

    # Plot adjacency matrix
    plot_adjacency(df, model_name, c, output_dir)
    
    ## Plot kernel basis
    plot_kernel_basis(df, model_name, c, output_dir)

    ## Plot inferred WEIGHTED kernel with uncertainty over last so many c.samples
    plot_weighted_impulse(df, model_name, c, output_dir)

    ## Plot trace of parameters
    plot_trace()

    plt.close('all')

##########################
## Grid simulation functions
##########################

def do_hawkes_grid_sim(grid_keys, grid_pars, df, coverage_var='MP_stories_total'):
    grid_configs = []
    grid_models = []
    grid_W = []

    for i, par in enumerate(grid_pars):
        print f'Starting experiment #{i} out of {len(grid_pars)}'
        thresh = par[grid_keys.index('threshold')]
        grid_configs += [{
            'timeseries': np.array([
                ((df['pms_count']>0) & (df['PMS_numkilled']<thresh)).astype(float).values,
                ((df['pms_count']>0) & (df['PMS_numkilled']>=thresh)).astype(float).values,
                df[coverage_var].values,
                ]).T.astype(int) # Make discrete
            ,
            'timeseries_labels': [
                f'PMS < {thresh}', 
                f'PMS >= {thresh}', 
                'Coverage'
                ],
            'time_shifts': [ 0, 0, 0, ],
            'hawkes_params': get_hawkes_params(**{k:v for k,v in zip(grid_keys, par) if k != 'threshold'})
        }]
        model_name = f'thresh{thresh} : {par}'
        grid_models += [util.do_pyhawkes_sim(grid_configs[-1], N_samples=1000, use_samples=250)]
        grid_W += [np.array([s.W for i, s in enumerate(grid_models[-1]['samples']) 
                        if i>= grid_models[-1]['N_samples']-grid_models[-1]['use_samples']])]

    return grid_configs, grid_models, np.array(grid_W)
