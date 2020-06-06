import numpy as np
import pandas as pd

## Read and format data
data_dir = 'data/'
df = pd.read_csv(data_dir + '2020-03-12_contagion_data.csv')

split_threshold = 6

def get_hawkes_params(dt=1, dt_max=30, B=20,
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


ts_dic = {
    'Shootings; MP normalized': {
        'timeseries': np.array([
            ((df['pms_count']>0)).values,
            df['MP_stories_total'].values / df['MP_stories_total'].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'PMS', 
            'MP Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=10)
    },
    'Shootings; AP normalized': {
        'timeseries': np.array([
            ((df['pms_count']>0)).values,
            df['AP_stories_total'].values / df['TVShoot_stories_total'].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'PMS', 
            'AP Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=10)
    },
    'Shootings; TV normalized': {
        'timeseries': np.array([
            ((df['pms_count']>0)).values,
            df['TVShoot_stories_total'].values / df['TVShoot_stories_total'].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            'PMS', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, ],
        'hawkes_params': get_hawkes_params(dt_max=30, B=10)
    },
        
        
        
    f'Shootings <>{split_threshold}; MP normalized': {
        'timeseries': np.array([
            ((df['pms_count']>0) & (df['PMS_numkilled']<split_threshold)).values,
            ((df['pms_count']>0) & (df['PMS_numkilled']>=split_threshold)).values,
            df['MP_stories_total'].values / df['MP_stories_total'].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'PMS < {split_threshold}', 
            f'PMS >= {split_threshold}', 
            'MP Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=10)
    },        
    f'Shootings <>{split_threshold}; AP normalized': {
        'timeseries': np.array([
            ((df['pms_count']>0) & (df['PMS_numkilled']<split_threshold)).values,
            ((df['pms_count']>0) & (df['PMS_numkilled']>=split_threshold)).values,
            df['AP_stories_total'].values / df['AP_stories_total'].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'PMS < {split_threshold}', 
            f'PMS >= {split_threshold}', 
            'AP Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=10)
    },        
    f'Shootings <>{split_threshold}; TV normalized': {
        'timeseries': np.array([
            ((df['pms_count']>0) & (df['PMS_numkilled']<split_threshold)).values,
            ((df['pms_count']>0) & (df['PMS_numkilled']>=split_threshold)).values,
            df['TVShoot_stories_total'].values / df['TVShoot_stories_total'].std(),
            ]).T.astype(int) # Make discrete
        ,
        'timeseries_labels': [
            f'PMS < {split_threshold}', 
            f'PMS >= {split_threshold}', 
            'TV Coverage'
            ],
        'time_shifts': [ 0, 0, 0,],
        'hawkes_params': get_hawkes_params(dt_max=30, B=10)
    },
}
