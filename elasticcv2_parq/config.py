from typing import Any, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import torch

# Allow utilization of tensor cores
torch.backends.cuda.matmul.allow_tf32 = True

class ElasticcConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='ELASTICCv2parquet_',
        env_file='.env',
        extra="ignore"
    )
    pretrained_model: Optional[str] = Field(None)
    eval_checkpoint: Optional[str] = Field(None)
    eval_batch_size: Optional[int] = Field(128)
    gaussian_noise: Optional[bool] = Field(False)
    model_size: str = Field("RoMAE-tiny")
    pretrain_epochs: int = Field(400)
    pretrain_lr: float = Field(4e-4)
    pretrain_warmup_steps: int = 20
    pretrain_batch_size: int = Field(350)
    pretrain_eval_every: int = Field(1000)
    pretrain_save_every: int = Field(50)
    pretrain_mask_ratio: float = Field(0.4)
    pretrain_grad_clip: float = Field(10)
    ## This will not finetune supposedly, reinstance carefully if we need to
    #finetune_epochs: int = Field(25)
    # finetune_lr: float = Field(1e-4)
    # finetune_warmup_steps: int = 2000
    # finetune_batch_size: int = Field(256)
    # finetune_eval_every: int = Field(500)
    # finetune_save_every: int = Field(500)
    # finetune_grad_clip: float = Field(10)
    # finetune_label_smoothing: float = Field(0.0)
    # finetune_use_class_weights: bool = Field(True)
    ## Mhhhh?
    # class_weights: list[float] = Field([
    #     0.2011, 1.2752, 0.7855, 0.6691, 1.6178, 0.2065, 1.4077, 2.4635, 7.7423,
    #     0.1898, 0.9508, 0.1902, 0.3751, 0.0944, 0.3725, 0.0644, 0.0449, 0.1929,
    #     0.1939, 0.7324
    # ])
    
    ## This changes because it's now elasticc2:
    n_classes: int = Field(32)
    
    # class_names: list[str] = Field([ 'AGN', 'CART', 'Cepheid', 'Delta Scuti', 'Dwarf Novae', 'EB', 'ILOT',
    #                           'KN', 'M-dwarf Flare', 'PISN', 'RR Lyrae', 'SLSN', '91bg', 'Ia', 'Iax', 'Ib/c',
    #                           'II', 'SN-like/Other', 'TDE', 'uLens'])

    #'EB', 'SNIa-SALT3', 'SNIc+HostXT_V19', 'uLens-Single-GenLens',
    # 'KN_K17', 'CLAGN', 'SLSN-I+host', 'CART', 'Cepheid',
    # 'SNIb-Templates', 'SNIIb+HostXT_V19', 'SNII+HostXT_V19',
    # 'Mdwarf-flare', 'd-Sct', 'uLens-Single_PyLIMA', 'SNII-NMF',
    # 'KN_B19', 'RRL', 'SLSN-I_no_host', 'ILOT', 'SNIb+HostXT_V19',
    # 'TDE', 'SNIax', 'SNIIn-MOSFIT', 'SNIa-91bg', 'SNIc-Templates',
    # 'SNIIn+HostXT_V19', 'SNIcBL+HostXT_V19', 'SNII-Templates',
    # 'dwarf-nova', 'PISN', 'uLens-Binary'
    # This might raise some issues because we don't have a label yet per class in the parquets but can be fixed later

    
    project_name: str = Field("Elasticcv2_parquet_")
    # finetune_optimargs: dict[str, Any] = {"betas": (0.9, 0.999),
    #                                       "weight_decay": 0.05}
    pretrain_optimargs: dict[str, Any] = {"betas": (0.9, 0.95),
                                          "weight_decay": 0.05}
