import numpy as np
import pandas as pd
from pybasicbayes.util.text import progprint_xrange
from types import SimpleNamespace
from copy import deepcopy
import logging, itertools

import pyhawkes
from pyhawkes.models import DiscreteTimeNetworkHawkesModelGammaMixture

from model_params import get_hawkes_params, var_names

import matplotlib.pyplot as plt


##########################
## Formatting conveniences
##########################

plt.ion()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
ls_cycle = lambda i: ['-', '--', '-.'][np.mod(i, 3)]
plot_format = '.png'

plot_dpi = 300

latex_template = r'''\documentclass[article]{{standalone}}
\usepackage{{booktabs}}
\begin{{document}}
{}
\end{{document}}
'''

def latex_float(f):
    """See https://stackoverflow.com/a/13490601"""
    if np.isnan(f): return '-'
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

##########################
## Statistics functions
##########################

def extract_impulses(c, p=[2.5, 50, 97.5]):
    """Extract the weighted impulse response function after the warmup samples
    for the model *c* and return percentile values *p* of the time distribution.
    """
    if type(c) is dict:
        c = SimpleNamespace(**c)
    return np.percentile(
        np.array([s.impulses * s.W for s in c.samples[-c.use_samples:]]), 
        p, axis=0)

def H_significance(c, i, j):
    """Calculate the signifiance of an element of a model's H matrix
    in terms of the distance between the median of the distribution
    and 0 in units of the 1 sigma range (84th - 16th percentile).
    """
    p = extract_impulses(c , p=[16, 50, 84])[:,:,i, j]
    return np.nanmax(p[1] / (p[2] - p[0]), axis=0)

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

def do_pyhawkes_sim(config, N_samples=2000, use_samples=1000, thinning=1,
                    pyhawkes_model=DiscreteTimeNetworkHawkesModelGammaMixture):
    fitted_model = {
        'N_samples':int(N_samples/thinning), 
        'use_samples': int(use_samples/thinning), 
        'thinning': int(thinning)}
    
    ## Gather data
    timeseries, timeseries_labels, time_shifts = \
        config['timeseries'], config['timeseries_labels'], config['time_shifts']
    ## Instantiate model
    test_model = pyhawkes_model(
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
    """Plot the probability trace of the MCMC fit to serve as a
    diagnostic of convergence.
    """
    plt.figure()
    plt.plot(np.arange(c.N_samples), c.lps, 'k')
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.axvline(c.N_samples - c.use_samples, ls='dashed', color='orange')
    plt.show()
    plt.savefig(output_dir + 'contagion_' + model_name + '_converge' + plot_format, dpi=plot_dpi)

def plot_zoom_rate(df, model_name, c, output_dir, plot_start=None, plot_delta=60):
    """Make a plot of the underlying count data and modeled rate function for
    a fited Hawkes process model, zoomed in around a particular time point.
    """
    if plot_start is None:
        ## Pick the event that generated the maximum amount of coverage
        plot_start = df[var_names['MP_stories']].argmax() - 3
    c.last_model.plot(T_slice=(plot_start-plot_delta, plot_start+plot_delta))
    plt.savefig(output_dir + 'contagion_' + model_name + '_plot_zoom_rate' + plot_format, dpi=plot_dpi)

def plot_adjacency(df, model_name, c, output_dir):
    """Plot the adjacency matrix of the Hawkes process network.
    """
    plt.figure()
    c.last_model.plot_adjacency_matrix()
    plt.savefig(output_dir + 'contagion_' + model_name + '_plot_adjacency' + plot_format, dpi=plot_dpi)

def plot_kernel_basis(df, model_name, c, output_dir):
    """Plot the basis functions of the Hawkes process kernel.
    """
    plt.figure()
    plt.plot(c.last_model.basis.create_basis())
    plt.axvline(c.max_basis_days, ls='dashed', label='Inferred trunctation position')
    plt.legend()
    plt.savefig(output_dir + 'contagion_' + model_name + '_plot_basis' + plot_format, dpi=plot_dpi)

def plot_weighted_impulse(df, model_name, c, output_dir, axs=None, fig_range=None):
    """Plot the fitted impulse functions of the Hawkes process network for
    each variable, adjusted for the relative weight of the network relation
    between each variable pair.
    """
    ## Calculate weighted impulse response after the warmup samples
    imps = extract_impulses(c)
    if fig_range is None: fig_range = range(imps.shape[-1])
    if axs is None: 
        fig, axs = plt.subplots(imps.shape[-1], sharex='all', sharey='all')
    else:
        fig = axs[fig_range[0]].get_figure()
    kernels = c.last_model.get_parameters()[2]
    if len(kernels) == 1: axs = [axs]
    for i in fig_range:
        ax = axs[i]
        for j in range(imps.shape[-1]):
            ax.plot(np.arange(len(imps[0,:,j,i])), imps[1,:,j,i], 
                        label=c.timeseries_labels[j], color=colors[j], zorder=1, lw=2,
                        ls=ls_cycle(j))
            ax.fill_between(np.arange(len(imps[0,:,j,i])), imps[0,:,j,i], imps[2,:,j,i], 
                                label=None, alpha=0.5, color=colors[j], zorder=0, lw=0)
        ax.set_title('Weighted Effect on\n'+c.timeseries_labels[i], ha='center')
    plt.figtext(0.0, 0.5, 'Impulse response ($H_{k\\prime~\\rightarrow~k}$)',
                va='center', rotation=90)
    ax.legend(title='Effect of...', bbox_to_anchor=(0.9, .7), prop={'size': 6})
    ax.set_xlabel('Days')
    plt.xlim(0, c.max_basis_days)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(
            output_dir + 'contagion_' + model_name + '_plot_impulses_95per_weight' + plot_format, 
            dpi=plot_dpi)

def plot_trace(df, model_name, c, output_dir):
    """Plot the MCMC parameter trace of the fitted model."""
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
    axs[0, 0].set_ylabel('W; weight matrix')
    axs[0, 0].axvline(c.use_samples, zorder=-1, ls='dashed', color='0.5', label='warmup')
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
    plt.savefig(output_dir + 'contagion_' + model_name + '_traces' + plot_format, dpi=plot_dpi)

def do_pyhawkes_plots(df, model_name, fitted_model, output_dir=''):
    """Generate a standardized set of plots for each fitted Hawkes
    process model.
    """
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
    plot_trace(df, model_name, c, output_dir)

    plt.close('all')

##########################
## Grid simulation functions
##########################

def do_hawkes_grid_sim(grid_keys, grid_pars, df, coverage_var=var_names['MP_stories'],
                       N_samples=1500, use_samples=1000, 
                       count_var=var_names['MPS_count'], fatal_var=var_names['MPS_fatal']):
    """Simulate Hawkes process modeling across a grid of configuration parameters.
    """
    grid_configs = []
    grid_models = []
    grid_W = []

    for i, par in enumerate(grid_pars):
        logging.info(f'Starting experiment #{i} out of {len(grid_pars)}')
        thresh = par[grid_keys.index('threshold')]
        grid_configs += [{
            'timeseries': np.array([
                ((df[count_var]>0) & (df[fatal_var]<thresh)).astype(float).values,
                ((df[count_var]>0) & (df[fatal_var]>=thresh)).astype(float).values,
                df[coverage_var].values,
                ]).T.astype(int) # Make discrete
            ,
            'timeseries_labels': [
                f'MPS (lower severity)', 
                f'MPS (higher severity)', 
                'Coverage'
                ],
            'time_shifts': [ 0, 0, 0, ],
            'hawkes_params': get_hawkes_params(**{k:v for k,v in zip(grid_keys, par) if k != 'threshold'})
        }]
        model_name = f'thresh{thresh} : {par}'
        grid_models += [do_pyhawkes_sim(grid_configs[-1], N_samples=N_samples, use_samples=use_samples)]
        grid_W += [np.array([s.W for i, s in enumerate(grid_models[-1]['samples']) 
                        if i>= grid_models[-1]['N_samples']-grid_models[-1]['use_samples']])]

    return grid_configs, grid_models, np.array(grid_W)

def plot_grid_averaged(grid_keys, grid_pars, grid_W, 
                       label_effect_of, label_effect_on, 
                       index_effect_of, index_effect_on, plot_dir):
    fig, axs = plt.subplots(1, len(grid_keys), sharey='all', figsize=(12,5))
    axs[0].set_ylabel('Weight ($W_{k\\prime~\\rightarrow~k}$) for:\neffect of' + 
                      f' {label_effect_of} on {label_effect_on}')
    for i, ax in enumerate(axs):
        pds = pd.Series(index=np.array(grid_pars)[:, i], 
                data=grid_W.mean(axis=1)[:, index_effect_of, index_effect_on])
        pds_g = pds.groupby(level=0).mean()
        pax = ax.plot(pds_g.index, pds_g.values, '-o')
        ax.set_xlabel(grid_keys[i])

    plt.tight_layout()
    plt.savefig(plot_dir+f'grid_search_model_results_simple_{index_effect_of}_{index_effect_on}' + plot_format, dpi=plot_dpi)

def plot_grid_iso(grid_keys, grid_pars, grid_W, 
                  label_effect_of, label_effect_on, 
                  index_effect_of, index_effect_on, plot_dir):
    fig, axs = plt.subplots(1, len(grid_keys), sharey='all', figsize=(12,5))
    axs[0].set_ylabel('Weight ($W_{k\\prime~\\rightarrow~k}$) for:\neffect of' + 
                      f' {label_effect_of} on {label_effect_on}')
    for i, ax in enumerate(axs):
        ## Split experiments by other parameters
        ## Remove the target column and then use np.unique to get the elements
        ## of each unique combination
        groups = np.unique(np.delete(grid_pars, i, axis=1), return_inverse=True, axis=0)[1]
        for group_i in np.unique(groups):
            sel = (groups == group_i)
            px = np.array(grid_pars)[sel, i].astype(float)
            ## jitter
            px += np.random.normal(0, (px[1:]-px[:-1]).mean()/20, len(px))
            py = np.percentile(grid_W[sel, :, index_effect_of, index_effect_on], 
                            [2.5, 50, 97.5], axis=1)
            pax = ax.errorbar(px, py[1], yerr=[py[1]-py[0], py[2]-py[1]], fmt='-o', color='k', alpha=0.5)
        ax.set_xlabel(grid_keys[i])

    plt.tight_layout()
    plt.savefig(plot_dir+f'grid_search_model_results_iso_{index_effect_of}_{index_effect_on}' + plot_format, dpi=plot_dpi)

def plot_grid_results(grid_keys, grid_configs, grid_pars, grid_W, plot_dir=''):
    K = grid_configs[0]['timeseries'].shape[1]
    for index_effect_of in range(K):
        for index_effect_on in range (index_effect_of, K):
            label_effect_of = grid_configs[-1]['timeseries_labels'][index_effect_of]
            label_effect_on = grid_configs[-1]['timeseries_labels'][index_effect_on]

            ## Simple plot of grid search results on K dimensions averaged over the other axes
            plot_grid_averaged(grid_keys, grid_pars, grid_W, 
                               label_effect_of, label_effect_on, 
                               index_effect_of, index_effect_on, plot_dir)
            
            ## Iso plot of grid search results on K dimensions -- 
            ## each line corresponds to constant values of the parameters not shown on that facet
            plot_grid_iso(grid_keys, grid_pars, grid_W, 
                          label_effect_of, label_effect_on, 
                          index_effect_of, index_effect_on, plot_dir)
            
        plt.close('all')


##########################
## Significance simulation functions
##########################

def do_hawkes_sig_sim(model_config,
        N_sim_iterations = 20,
        N_sim_sizes = np.array([5, 10, 20, 40, 80]) * 365,
        W_vals = np.array([1e-5, 1e-2, 1e-1, .25, .5, .7]),
        sim_use_samples = 100,
        sim_MCMC_samples = 600,
        target_node = (0,0)
        ):
    """
    Run a simulation where a single adjacency matrix value of the base_model is 
    systematically adjusted, fake data is generated under that model, and then
    a new model is fit to the fake data.
    
    NOTE: Running this simulation will take of order a few minutes per N_sim_iterations.
    
    Parameters
    ---------
    model_config: dict
        Model configuration to pass to do_pyhawkes_sim
    N_sim_iterations: int
        Number of iterations to run for each configuration, sample size, and W value.
    N_sim_sizes: List[int]
        Number of days worth of data to generate in each simulation
    W_vals: list[float]
        Values of W to systematically test.
    sim_use_samples: int
        The number of MCMC samples to use (post-warmup) from each MCMC chain.
    sim_MCMC_samples: int
        The number of MCMC samples to run when doing inference on each simulated dataset.
    target_node: tuple[int]
        The (i,j) coordinates of the node of the adjacency matrix to manipulate.
    
    Returns
    -------
    sim_models: dict[dict[list(pyhawkes model dict)]]
        A set of model fits to simulated data organized by 
        W_vals[N_sim_sizes[N_sim_iterations]]
    """
    ## Fit a fresh model to the real data
    c_e = do_pyhawkes_sim(model_config)
    ## Iterate over sample sizes
    sim_models = {}
    for Wzz in W_vals:
        sim_models[Wzz] = {}
        for N_sim in N_sim_sizes:
            sim_models[Wzz][N_sim] = []
            ## Iterate over simulation iterations
            for i in range(N_sim_iterations):
                ## Pick a random model
                i_pick = np.random.randint(len(c_e['samples'])-c_e['use_samples'], len(c_e['samples']))
                c_e_pick = deepcopy(c_e['samples'][i_pick])
                c_e_pick.W[target_node] = Wzz
                ## Generate N_sim data samples
                gen_samples = c_e_pick.generate(T=int(N_sim), keep=False)[0]
                ## Fit a fresh model to the generated data
                model_config = deepcopy(model_config)
                model_config['timeseries'] = gen_samples
                sim_models[Wzz][N_sim] += [do_pyhawkes_sim(model_config, 
                            N_samples = sim_MCMC_samples, use_samples=sim_use_samples)]
    return sim_models

def plot_sig_sim_results(sim_models, target_node=(0,0),
                actual_data_length = None):
    """Plot the results of a do_hawkes_sig_sim simulation.  The input should
    be the dictionary of model results returned by do_hawkes_sig_sim.
    """
    N_sim_sizes = np.array(list(sim_models[list(sim_models.keys())[0]].keys()))
    ## Plot sample size versus significance
    sim_significance = {Wzz: {N_sim: [H_significance(sim_models[Wzz][N_sim][i], 
                                            target_node[0], target_node[1]) 
                            for i in range(len(sim_models[Wzz][N_sim]))]
                                for N_sim in sim_models[Wzz]}
                                    for Wzz in sim_models}

    plt.figure()
    sim_significance_p = np.array([[
                    np.percentile(sim_significance[Wzz][N_sim], [16, 50, 84])
                            for N_sim in sim_significance[Wzz]]
                                for Wzz in sim_significance]).T
    for Wi, Wzz in enumerate(sim_significance):
        color = plt.cm.viridis(float(Wi) / len(sim_significance))
        ssp = sim_significance_p[:,:,Wi]
        ## Plot the 1sigma errors
        plt.errorbar(N_sim_sizes / 365 + Wi/2, ssp[1], 
                    [ssp[1] - ssp[0], ssp[2] - ssp[1]], 
                    fmt='-o', lw=1, color=color, 
                    label=np.format_float_positional(Wzz, trim='-'))
    plt.axhline(1, ls='dashed')
    if actual_data_length is not None:
        plt.axvline(actual_data_length, 
                ls='dotted', color='0.5', label='Actual sample')
    plt.xlabel('Amount of simulated data (years)')
    plt.ylabel('Significance of inferred\nshooting self-excitation ($S_{H, \\rm{max}}$)')
    plt.legend(title='Adjacency weight $W_{0\\rightarrow0}$', bbox_to_anchor=(1,1))
    #plt.semilogy()
    plt.tight_layout()

