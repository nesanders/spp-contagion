import numpy as np
import pandas as pd

## Read and format data
data_dir = 'data/'
df = pd.read_csv(data_dir + '2020-07-17_contagion_data.csv')
## Variable --> column name lookup
var_names = {
    'numday': 'numday',
    'pop': 'Population',  ## TODO population is unused
    'MPS_count': 'MPS_incs',
    'MPS_fatal': 'MPS_vics',
    'MS_count': 'MS_incs',
    'MS_fatal': 'MS_vics',
    'MP_stories': 'MP_stories',
    'AP_stories': 'AP_stories',
    'TV_stories': 'TVSH_stories'
    }

split_threshold = 6

def get_hawkes_params(dt=1, dt_max=30, B=16,
                      basis_allow_instantaneous=False,
                      basis_norm=True,
                      bkgd_alpha=1.0, bkgd_beta=1.0,
                      impulse_gamma=1.0,
                      weight_hypers={},
                      #network_C=4,
                      network_allow_self_connections=True,
                      network_alpha=None, network_beta=None, 
                      #network_c=np.array([0,1]), 
                      network_kappa=1.0, 
                      #network_pi=1.0, 
                      network_v=None,
                      network_p=0.5
                      ):
    return {
        'dt': dt, # Timestep for basis function
        'dt_max': dt_max, # truncate basis function at
        'B': B, # number of basis functions
        'basis_hypers': {'allow_instantaneous': basis_allow_instantaneous, 'norm': basis_norm},
        'bkgd_hypers': {'alpha': bkgd_alpha, 'beta': bkgd_beta},
        'impulse_hypers': {'gamma': impulse_gamma},
        'weight_hypers': weight_hypers,
        'network_hypers': {
            'allow_self_connections': network_allow_self_connections,
            'alpha': network_alpha, # Sets the offset of the shape parameter for the gamma scale of the weight matrix (v)
            'beta': network_beta, # Sets the offset of the scale parameter for the gamma scale of the weight matrix (v).  Note you're not allowed to set beta without setting alpha.
            'kappa': network_kappa, # Sets the shape of the weight matrix
            'v': network_v, ## Note - setting v overrides alpha and beta
            'p': network_p,  ## Scale of the KxK matrix of probabilities.  Should be >0 and <=1; default=0.5
        }
    }

## Configurations for Mass Public Shootings
ts_dic_MPS = {
    'MPS; MP normalized': {
        'timeseries': np.array([
            ((df[var_names['MPS_count']]>0)).values,
            df[var_names['MP_stories']].values / df[var_names['MP_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MPS', 
            'MP Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
    'MPS; AP normalized': {
        'timeseries': np.array([
            ((df[var_names['MPS_count']]>0)).values,
            df[var_names['AP_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MPS', 
            'AP Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
    'MPS; TV normalized': {
        'timeseries': np.array([
            ((df[var_names['MPS_count']]>0)).values,
            df[var_names['TV_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MPS', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
    
    'MPS; long timescale; TV normalized': {
        'timeseries': np.array([
            ((df[var_names['MPS_count']]>0)).values,
            df[var_names['TV_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MPS', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=180, B=16)
    },        
        
        
    f'MPS <>{split_threshold}; MP normalized': {
        'timeseries': np.array([
            ((df[var_names['MPS_count']]>0) & (df[var_names['MPS_fatal']]<split_threshold)).values,
            ((df[var_names['MPS_count']]>0) & (df[var_names['MPS_fatal']]>=split_threshold)).values,
            df[var_names['MP_stories']].values / df[var_names['MP_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'MPS < {split_threshold}', 
            f'MPS >= {split_threshold}', 
            'MP Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },        
    f'MPS <>{split_threshold}; AP normalized': {
        'timeseries': np.array([
            ((df[var_names['MPS_count']]>0) & (df[var_names['MPS_fatal']]<split_threshold)).values,
            ((df[var_names['MPS_count']]>0) & (df[var_names['MPS_fatal']]>=split_threshold)).values,
            df[var_names['AP_stories']].values / df[var_names['AP_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'MPS < {split_threshold}', 
            f'MPS >= {split_threshold}', 
            'AP Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },        
    f'MPS <>{split_threshold}; TV normalized': {
        'timeseries': np.array([
            ((df[var_names['MPS_count']]>0) & (df[var_names['MPS_fatal']]<split_threshold)).values,
            ((df[var_names['MPS_count']]>0) & (df[var_names['MPS_fatal']]>=split_threshold)).values,
            df[var_names['TV_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'MPS < {split_threshold}', 
            f'MPS >= {split_threshold}', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
}


## Configurations for Mass Shootings (not MPS)
ts_dic_MS = {
    'MS; MP normalized': {
        'timeseries': np.array([
            ((df[var_names['MS_count']]>0)).values,
            df[var_names['MP_stories']].values / df[var_names['MP_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MS', 
            'MP Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
    'MS; AP normalized': {
        'timeseries': np.array([
            ((df[var_names['MS_count']]>0)).values,
            df[var_names['AP_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MS', 
            'AP Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
    'MS; TV normalized': {
        'timeseries': np.array([
            ((df[var_names['MS_count']]>0)).values,
            df[var_names['TV_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MS', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
        
    'MS; long timescale; TV normalized': {
        'timeseries': np.array([
            ((df[var_names['MS_count']]>0)).values,
            df[var_names['TV_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'MS', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=180, B=16)
    },        
        
    f'MS <>{split_threshold}; MP normalized': {
        'timeseries': np.array([
            ((df[var_names['MS_count']]>0) & (df[var_names['MS_fatal']]<split_threshold)).values,
            ((df[var_names['MS_count']]>0) & (df[var_names['MS_fatal']]>=split_threshold)).values,
            df[var_names['MP_stories']].values / df[var_names['MP_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'MS < {split_threshold}', 
            f'MS >= {split_threshold}', 
            'MP Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },        
    f'MS <>{split_threshold}; AP normalized': {
        'timeseries': np.array([
            ((df[var_names['MS_count']]>0) & (df[var_names['MS_fatal']]<split_threshold)).values,
            ((df[var_names['MS_count']]>0) & (df[var_names['MS_fatal']]>=split_threshold)).values,
            df[var_names['AP_stories']].values / df[var_names['AP_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'MS < {split_threshold}', 
            f'MS >= {split_threshold}', 
            'AP Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },        
    f'MS <>{split_threshold}; TV normalized': {
        'timeseries': np.array([
            ((df[var_names['MS_count']]>0) & (df[var_names['MS_fatal']]<split_threshold)).values,
            ((df[var_names['MS_count']]>0) & (df[var_names['MS_fatal']]>=split_threshold)).values,
            df[var_names['TV_stories']].values / df[var_names['TV_stories']].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'MS < {split_threshold}', 
            f'MS >= {split_threshold}', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=16)
    },
}
