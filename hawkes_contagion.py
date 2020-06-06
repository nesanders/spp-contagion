## TODO rerun results with AP instead of MP and just add a paragraph explaining if the results look the same
## TODO simplify figures
## Calculate ratio of background to excitory rates liek Mohler
## TODO make figures grayscale friendly
## TODO add summary statistics about model predictive performance for coverage vs shootings
## TODO write up including equations from Linderman & Adams.  Overall narrative is that we can adaquately explain coverage based on excitation from high severity events, but the shootings are distributed randomly.
## TODO rerun everything with shorter time horizon

import numpy as np
import shutil, joblib, logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import contagion_pyhawkes_util as util
from model_params import data_dir, df, get_hawkes_params, ts_dic

logging.getLogger().setLevel(logging.INFO)
## Setup output directory
plot_dir = 'plots/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)


##########################
## Individual model fits
##########################

## Fit individual models and generate plots 
pyhawkes_models = {}
for model_name in list(ts_dic.keys()):
    pyhawkes_models[model_name] = util.do_pyhawkes_sim(ts_dic[model_name])
    util.do_pyhawkes_plots(df, model_name, pyhawkes_models[model_name], plot_dir = plot_dir)

## Save out results
joblib.dump(pyhawkes_models, data_dir+'pyhawkes_models.jl')


##########################
## Grid simulation
##########################

## Run models over a grid of parameters
grid_par_dic = {
    'threshold': np.arange(5, 21, 2),
    'network_v': [1, 2, 3, 6, 9],
    }
grid_keys = list(grid_par_dic.keys())
grid_pars = list(itertools.product(*grid_par_dic.values()))
## Do grid simulation
grid_configs, grid_models, grid_W = util.do_hawkes_grid_sim(grid_keys, grid_pars, df)
## Save out results
joblib.dump({
                'grid_W': grid_W, 
                'grid_par_dic':grid_par_dic, 
                'grid_configs':grid_configs,
                'grid_pars': grid_pars,
             }, data_dir+'grid_search_model_results.jl')
K = grid_configs[0]['timeseries'].shape[1]
for index_effect_of in range(K):
    for index_effect_on in range (index_effect_of, K):
        label_effect_of = grid_configs[-1]['timeseries_labels'][index_effect_of]
        label_effect_on = grid_configs[-1]['timeseries_labels'][index_effect_on]

        ## Plot Grid search results on K dimensions
        fig, axs = plt.subplots(1, len(grid_keys), sharey='all', figsize=(12,5))
        axs[0].set_ylabel(f'Adjacency weight (W) for:\neffect of {label_effect_of} on {label_effect_on}')
        for i, ax in enumerate(axs):
            px = np.array(grid_pars)[:, i]
            py = grid_W.mean(axis=1)[:, index_effect_of, index_effect_on]
            c_index = np.mod(i+1, len(grid_keys))
            pc = np.array(grid_pars)[:, c_index]
            s_index = np.mod(i+2, len(grid_keys))
            ps = np.array(grid_pars)[:, s_index]
            pax = ax.scatter(px, py, c=pc, s=ps, cmap=plt.cm.Reds)
            ax.set_xlabel(grid_keys[i])
            plt.colorbar(pax, label=grid_keys[c_index], ax=ax)
            ## Scatter legend
            s_handles, s_labels = pax.legend_elements(prop="sizes", alpha=0.6)
            ax.legend(s_handles, s_labels, loc="upper right", title=grid_keys[s_index])

        plt.tight_layout()
        plt.savefig(plot_dir+f'grid_search_model_results_multi_{index_effect_of}_{index_effect_on}.png')

        ## Simple plot of grid search results on K dimensions
        fig, axs = plt.subplots(1, len(grid_keys), sharey='all', figsize=(12,5))
        axs[0].set_ylabel(f'Adjacency weight (W) for:\neffect of {label_effect_of} on {label_effect_on}')
        for i, ax in enumerate(axs):
            pds = pd.Series(index=np.array(grid_pars)[:, i], 
                    data=grid_W.mean(axis=1)[:, index_effect_of, index_effect_on])
            pds_g = pds.groupby(level=0).mean()
            pax = ax.plot(pds_g.index, pds_g.values, '-o')
            ax.set_xlabel(grid_keys[i])

        plt.tight_layout()
        plt.savefig(plot_dir+f'grid_search_model_results_simple_{index_effect_of}_{index_effect_on}.png')
        
        
        ## Iso plot of grid search results on K dimensions -- 
        ## each line corresponds to constant values of the parameters not shown on that facet
        fig, axs = plt.subplots(1, len(grid_keys), sharey='all', figsize=(12,5))
        axs[0].set_ylabel(f'Adjacency weight (W) for:\neffect of {label_effect_of} on {label_effect_on}')
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
                                   [5, 50, 95], axis=1)
                pax = ax.errorbar(px, py[1], yerr=[py[1]-py[0], py[2]-py[1]], fmt='-o', color='k', alpha=0.5)
            ax.set_xlabel(grid_keys[i])

        plt.tight_layout()
        plt.savefig(plot_dir+f'grid_search_model_results_iso_{index_effect_of}_{index_effect_on}.png')

## TODO plot W ratio for two shooting variables

