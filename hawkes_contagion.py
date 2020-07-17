import numpy as np
import shutil, joblib, logging, itertools
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import util
from model_params import data_dir, df, get_hawkes_params, ts_dic_MPS, ts_dic_MS, split_threshold

logging.getLogger().setLevel(logging.INFO)
## Setup output directory
plot_dir = 'plots/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)


##########################
## Individual model fits
##########################

## Fit individual models and generate plots 
pyhawkes_models = {}
ts_dic_all = deepcopy(ts_dic_MPS); ts_dic_all.update(ts_dic_MS)
for model_name in ts_dic_all.keys():
    ## Fit model
    pyhawkes_models[model_name] = util.do_pyhawkes_sim(ts_dic_all[model_name])
    ## Make plots
    util.do_pyhawkes_plots(df, model_name, pyhawkes_models[model_name], output_dir = plot_dir)

## Plot comparison across 2-variable models
fig, axs = plt.subplots(2, 3, sharex='all', sharey='all', figsize=(8, 6))
for mi, model_name in enumerate([
                   'Shootings; MP normalized', 
                   'Shootings; AP normalized', 
                   'Shootings; TV normalized']):
    util.plot_weighted_impulse(df, model_name, 
                    util.SimpleNamespace(**pyhawkes_models[model_name]), 
                    plot_dir, axs=axs[:, mi])

#for ax in axs[1, 1:].flatten(): ax.get_legend().set_visible(False)
for ax in axs.flatten(): ax.set_title('on\n'.join(ax.get_title().split('on ')), ha='center')
plt.savefig(plot_dir + 'contagion_compare_nothresh_impulses_95per_weight' + util.plot_format,
            dpi=util.plot_dpi)

## Plot comparison across 3-variable models
fig, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(8, 4))
for mi, model_name in enumerate([
                    'Shootings <>6; MP normalized', 
                    'Shootings <>6; AP normalized', 
                    'Shootings <>6; TV normalized']):
    util.plot_weighted_impulse(df, model_name, 
                    util.SimpleNamespace(**pyhawkes_models[model_name]), 
                    plot_dir, axs=[None, None, axs[mi]], fig_range=[2])

for ax in axs[1:].flatten(): ax.get_legend().set_visible(False)
for ax in axs.flatten(): ax.set_title('on\n'.join(ax.get_title().split('on ')), ha='center')
plt.savefig(plot_dir + f'contagion_compare_thresh{split_threshold}_impulses_95per_weight' + util.plot_format,
            dpi=util.plot_dpi)

## Plot a selected zoom in
pyhawkes_models['Shootings; MP normalized']['last_model'].plot(
    T_slice=(5640, 5700))
fig = plt.gcf()
axs = fig.get_axes()
## Turn off network plot
axs[0].set_visible(False)
axs[1].set_ylabel('MPS')
axs[2].set_ylabel('MP coverage')
axs[-1].set_xlabel('')
for i in range(1, 2+1):
    axs[i].text(axs[i].get_xlim()[-1] - 2, axs[i].get_ylim()[1]* 0.05, 
                '$\lambda(t)$', ha='right', color='#377eb8')
axs[-1].set_xlabel('')  
xt = [np.datetime64(df.date.iloc[0]) + np.timedelta64(int(c), 'D') for c in axs[2].xaxis.get_ticklocs()]
axs[2].xaxis.set_ticklabels(xt, rotation=90)
plt.tight_layout()
plt.savefig(plot_dir + 'contagion_selected_plot_zoom_rate' + util.plot_format,
            dpi=util.plot_dpi)


##########################
## Grid simulation
##########################

## Run models over a grid of parameters
grid_par_dic = {
    'threshold': list(range(4,10)) + list(range(11,25,3)),
    'network_v': [1, 3, 6, 9],
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
## Plot results
util.plot_grid_results(['Fatality threshold', 'Weight concentration hyperparmeter ($\\nu$)'], 
                       grid_configs, grid_pars, grid_W, plot_dir=plot_dir)

