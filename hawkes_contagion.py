import numpy as np
import shutil
import os
import joblib
import logging
import itertools
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

ts_dic_all = deepcopy(ts_dic_MPS); ts_dic_all.update(ts_dic_MS)

##########################
## Individual model fits
##########################

## Fit individual models and generate plots 
pyhawkes_models = {}
for model_name in ts_dic_all.keys():
    if model_name not in pyhawkes_models: ## DEBUG - do not repeat models already fit
        ## Fit model
        pyhawkes_models[model_name] = util.do_pyhawkes_sim(ts_dic_all[model_name])
        ## Make plots
        util.do_pyhawkes_plots(df, model_name, pyhawkes_models[model_name], output_dir = plot_dir)

## Save out results -- TODO this is not working due to a serialization error
#joblib.dump(pyhawkes_models, data_dir+'individual_model_results.jl')

##########################
## Individual model fit plots
##########################

## Plot comparison across 2-variable models for MPS and MS
for s_type in ['MPS', 'MS']:
    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all', figsize=(8, 6))
    for mi, model_name in enumerate([
                    f'{s_type}; MP normalized', 
                    f'{s_type}; AP normalized', 
                    f'{s_type}; TV normalized']):
        util.plot_weighted_impulse(df, model_name, 
                        util.SimpleNamespace(**pyhawkes_models[model_name]), 
                        None, axs=axs[:, mi])

    #for ax in axs[1, 1:].flatten(): ax.get_legend().set_visible(False)
    for ax in axs.flatten(): ax.set_title('on\n'.join(ax.get_title().split('on ')), ha='center')
    plt.savefig(plot_dir + f'contagion_compare_{s_type}_nothresh_impulses_95per_weight' + util.plot_format,
                dpi=util.plot_dpi)

## Plot comparison across 3-variable models
fig, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(8, 4))
for mi, model_name in enumerate([
                    'MPS <>6; MP normalized', 
                    'MPS <>6; AP normalized', 
                    'MPS <>6; TV normalized']):
    util.plot_weighted_impulse(df, model_name, 
                    util.SimpleNamespace(**pyhawkes_models[model_name]), 
                    None, axs=[None, None, axs[mi]], fig_range=[2])

for ax in axs[1:].flatten(): ax.get_legend().set_visible(False)
for ax in axs.flatten(): ax.set_title('on\n'.join(ax.get_title().split('on ')), ha='center')
plt.savefig(plot_dir + f'contagion_compare_thresh{split_threshold}_impulses_95per_weight' + util.plot_format,
            dpi=util.plot_dpi)

## Plot a selected zoom in
pyhawkes_models['MPS; MP normalized']['last_model'].plot(
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
## Individual model fit table
##########################

ir_table_dic = {}
col_vars = ['Model type', 'Statistic', 'Shooting type', 'Media type', 'Timescale']
for model_name in pyhawkes_models:
    ## Setup column names
    m_stype = 'MPS' if 'MPS' in model_name else 'MS'
    m_mtype = model_name.split('; ')[-1].split(' norm')[0]
    m_timescale = 'Short' if 'long timescale' not in model_name else 'Long'
    if '<>' in model_name:
        m_var = 'Three variable'
        matrix_dims = ['Low severity shoot.', 'High severity shoot.', 'Coverage']
    else:
        m_var = 'Two variable'
        matrix_dims = ['Shootings', 'Coverage']
    ## Extract model
    c = util.SimpleNamespace(**pyhawkes_models[model_name])
    ## Calculate H max
    col_tuple = (m_var, r'$H_{\rm{max}}$', m_stype, m_mtype, m_timescale)
    ir_table_dic[col_tuple] = pd.DataFrame(data=util.extract_impulses( c, p=97.5).max(axis=0),
                                           index=matrix_dims, columns=matrix_dims)
    ## Calculate significance
    col_tuple = (m_var, r'$S_{H, \rm{max}}$', m_stype, m_mtype, m_timescale)
    ir_table_dic[col_tuple] = pd.DataFrame(util.H_significance(c, slice(None), slice(None)),
                                           index=matrix_dims, columns=matrix_dims)

ir_table_df = pd.concat(ir_table_dic, names=col_vars).sort_index(level=[0,1,2,3, 4])
ir_table_df.index.set_names('Effect of...', level=-1, inplace=True)
ir_table_df.columns.set_names(['Effect on...'], inplace=True)
## Save a latex version
with open('data/impulse_response_table.tex', 'w') as f:
    f.write(util.latex_template.format(ir_table_df.to_latex(escape=False,
                            float_format=util.latex_float)))
## PDF it
os.system('pdflatex data/impulse_response_table.tex data/impulse_response_table.pdf')
## Also save a csv version
ir_table_df.to_csv('data/impulse_response_table.csv')


##########################
## Sample size simulations
##########################

target_node = (0,0)
sim_sig_output = util.do_hawkes_sig_sim(ts_dic_all['MPS; MP normalized'])
util.plot_sig_sim_results(sim_sig_output, 
            actual_data_length = len(ts_dic_all[model_name]['timeseries']) / 365,
            target_node=target_node)
plt.savefig(plot_dir + f'significance_simulation_H{target_node}' + util.plot_format,
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

