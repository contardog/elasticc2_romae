"""
A simple example dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from romae.utils import gen_mask
#import joblib
import polars as pl

## We need to create a padded version of the full parquet file

def padd_parquet(parqu_, col_names_to_pad=['FLUXCAL', 'FLUXCALERR', 'MJD', 'BAND']):
    ##  
    #parqu_ = pl.read_parquet('/scratch/gcontard/ELASTICC2/combined_train_parquets/train.parquet')

    ## Find max length
    #lents = [len(p) for p in parqu_['FLUXCAL']]
    maxlen = max(parqu_['FLUXCAL'].list.len()) #max(lents)
    print(f"Max lengths of parquet {maxlen}")
    padd_mask = False
    for col in col_names_to_pad:
        
        parqu_ = parqu_.with_columns((pl.col(col)).alias(col+"_pad"))
        
        parqu_ = parqu_.with_columns(
           pl.col(col+"_pad").list.concat(
              pl.lit(0).repeat_by(
                 maxlen - pl.col(col).list.len()
              )
           )
        )
        
        ## ADD A TRACK OF THE PADD MASK  
        if not(padd_mask):
            
            parqu_ = parqu_.with_columns(
                pl.lit(False).repeat_by(
                            pl.col(col).list.len()).list.concat(pl.lit(True).repeat_by(maxlen - pl.col(col).list.len())).alias("PADD_MASK"))
            padd_mask = True
                
    ## I thought Array would be better once we padded everything but i must using this wrong.
    ## Also I'd like to fucking keep it f32 but?! why no?! let's see if this is a big issue down the line...

    ## I DONT KNOW IF WE SHOULD INCLUDE FLAGS AND ALL , ARE THERE ANY IN ELASTICC2 THAT SIM LSST?
    
    return parqu_

def map_bands(band_letters):
    band_dic = {
        'u': 0,
        'g': 1,
        'r': 2,
        'i': 3,
        'z': 4,
        'Y': 5,
        'y':5,
        "0" :-1
    }
    return [band_dic[l] for l in band_letters]


def map_labels(label_str):
    label_dic = {
        'CART':0, 
        'CLAGN':1,
        'Cepheid':2,
        'EB':3, 
        'ILOT':4,
        'KN_B19':5,
        'KN_K17':6,
        'Mdwarf-flare':7, 
        'PISN':8, 
        'RRL':9, 
        'SLSN-I+host':10,
        'SLSN-I_no_host':11,
        'SNII+HostXT_V19':12, 
        'SNII-NMF':13,
        'SNII-Templates':14,
        'SNIIb+HostXT_V19':15, 
        'SNIIn+HostXT_V19':16, 
        'SNIIn-MOSFIT':17,
        'SNIa-91bg':18, 
        'SNIa-SALT3':19,
        'SNIax':20, 
        'SNIb+HostXT_V19':21,
        'SNIb-Templates':22,
        'SNIc+HostXT_V19':23, 
        'SNIc-Templates':24,
        'SNIcBL+HostXT_V19':25,
        'TDE':26, 
        'd-Sct':27, 
        'dwarf-nova':28, 
        'uLens-Binary':29,
        'uLens-Single-GenLens':30,
        'uLens-Single_PyLIMA':31
    }
    return [label_dic[l] for l in label_str]

def reformat_bands(parqu_):
    ## This reformats the bands from letters to numbers
    ## Force return_dtype list of int32 instead of 64? 
    parqu_ = parqu_.with_columns(pl.col("BAND_pad").map_elements(map_bands, return_dtype=list[int]).alias("band_number"))
        
    return parqu_


def reformat_labels(parqu_):
    parqu_ = parqu_.with_columns(pl.col("ELASTICC_class").map_elements(map_labels, return_dtype=list[int]).alias("ELASTICC_class_int"))
        
    return parqu_

    
class ElasticcParquetDataset(Dataset):
    """
    This assumes some naming in the parquet taken as input -- maybe change to something better when we also adjust for DP1?
    """
    

    def __init__(self, parquet_file, 
                 mask_ratio: float = 0.5, gaussian_noise: bool = False):
        
        self.noise = gaussian_noise
        
        self.mask_ratio = mask_ratio

        #if isinstance(parquet_input, str):
        self.parquet = pl.read_parquet(parquet_file)
        # else:
        #     self.parquet = parquet_file
        
        
        self.parquet = padd_parquet(self.parquet)
        self.parquet = reformat_bands(self.parquet)
    ## Hopefully we don't need that?
    # def get_standardization_vals(self):
    #     import tqdm
    #     n_samples = self.file["data"].shape[0]
    #     means = torch.zeros(6)
    #     stds = torch.zeros(6)
    #     for i in tqdm.tqdm(range(n_samples), total=n_samples):
    #         data = self.file["data"][i]
    #         mask = self.file["mask"][i]
    #         for j in range(6):
    #             means[j] += data[:, j][mask[:, j] > 0.5].mean() / n_samples
    #             stds[j] += data[:, j][mask[:, j] > 0.5].std() / n_samples

    #     return means, stds


    
    

    def __len__(self):
        return len(self.parquet)

    def __enter__(self):
        return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.file.close()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.parquet

    def __getitem__(self, idx):
        
        # FLuxCal here should be the DIFF flux !!
        data = torch.tensor(self.parquet["FLUXCAL_pad"][idx].to_numpy()).flatten()
        pad_mask = ~(torch.tensor(self.parquet["PADD_MASK"][idx].to_numpy())).flatten()
        # Adjust if we have alert and flags?
        # alert_mask = (torch.tensor(self.file["mask_alert"][idx]) > 0.5).flatten()
        # pad_mask[alert_mask] = True
        times = torch.tensor(self.parquet["MJD_pad"][idx].to_numpy().flatten())
        times[pad_mask] = times[pad_mask] - torch.min(times[pad_mask]) #To avoid big numbers in times
        
        #label =  self.parquet["ELASTICC_class"][idx] # THIS IS A STRING 
        bands = torch.tensor(self.parquet["band_number"][idx].to_numpy()) # this is padded
        positions = torch.stack([bands, times])
        data_var = torch.tensor(self.parquet["FLUXCALERR_pad"][idx]).flatten()
        data = torch.stack([data, data_var])
        n_nonpad = pad_mask.sum()
        positions = nn.functional.pad(positions[:, pad_mask], (0, positions.shape[1]-n_nonpad)).float()
        data = nn.functional.pad(data[:, pad_mask], (0, data.shape[1]-n_nonpad))[..., None, None].float().swapaxes(0, 1)
        pad_mask[:] = False
        pad_mask[n_nonpad:] = True
        mask = gen_mask(self.mask_ratio, pad_mask[None, ...], single=True).squeeze()
        if self.noise:
            data = data + torch.randn_like(data) * 0.02
        sample = {
            "values": data,
            "positions": positions,
            #"label": label,
            "mask": mask,
            "pad_mask": pad_mask
        }
        return sample


  
class ElasticcParquetDatasetwLabel(Dataset):
    """
    This assumes some naming in the parquet taken as input -- maybe change to something better when we also adjust for DP1?
    """
    

    def __init__(self, parquet_file, 
                 mask_ratio: float = 0.5, gaussian_noise: bool = False):
        
        self.noise = gaussian_noise
        
        self.mask_ratio = mask_ratio

        #if isinstance(parquet_input, str):
        self.parquet = pl.read_parquet(parquet_file)
        # else:
        #     self.parquet = parquet_file
        
        
        self.parquet = padd_parquet(self.parquet)
        self.parquet = reformat_bands(self.parquet)

        self.parquet = reformat_labels(self.parquet)
    ## Hopefully we don't need that?
    # def get_standardization_vals(self):
    #     import tqdm
    #     n_samples = self.file["data"].shape[0]
    #     means = torch.zeros(6)
    #     stds = torch.zeros(6)
    #     for i in tqdm.tqdm(range(n_samples), total=n_samples):
    #         data = self.file["data"][i]
    #         mask = self.file["mask"][i]
    #         for j in range(6):
    #             means[j] += data[:, j][mask[:, j] > 0.5].mean() / n_samples
    #             stds[j] += data[:, j][mask[:, j] > 0.5].std() / n_samples

    #     return means, stds


    
    

    def __len__(self):
        return len(self.parquet)

    def __enter__(self):
        return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.file.close()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.parquet

    def __getitem__(self, idx):
        
        # FLuxCal here should be the DIFF flux !!
        data = torch.tensor(self.parquet["FLUXCAL_pad"][idx].to_numpy()).flatten()
        pad_mask = ~(torch.tensor(self.parquet["PADD_MASK"][idx].to_numpy())).flatten()
        # Adjust if we have alert and flags?
        # alert_mask = (torch.tensor(self.file["mask_alert"][idx]) > 0.5).flatten()
        # pad_mask[alert_mask] = True
        times = torch.tensor(self.parquet["MJD_pad"][idx].to_numpy().flatten())
        times[pad_mask] = times[pad_mask] - torch.min(times[pad_mask]) #To avoid big numbers in times
        
        label =  self.parquet["ELASTICC_class_int"][idx] #
        bands = torch.tensor(self.parquet["band_number"][idx].to_numpy()) # this is padded
        positions = torch.stack([bands, times])
        data_var = torch.tensor(self.parquet["FLUXCALERR_pad"][idx]).flatten()
        data = torch.stack([data, data_var])
        n_nonpad = pad_mask.sum()
        positions = nn.functional.pad(positions[:, pad_mask], (0, positions.shape[1]-n_nonpad)).float()
        data = nn.functional.pad(data[:, pad_mask], (0, data.shape[1]-n_nonpad))[..., None, None].float().swapaxes(0, 1)
        pad_mask[:] = False
        pad_mask[n_nonpad:] = True
        mask = gen_mask(self.mask_ratio, pad_mask[None, ...], single=True).squeeze()
        if self.noise:
            data = data + torch.randn_like(data) * 0.02
        sample = {
            "values": data,
            "positions": positions,
            "label": label,
            "mask": mask,
            "pad_mask": pad_mask
        }
        return sample

