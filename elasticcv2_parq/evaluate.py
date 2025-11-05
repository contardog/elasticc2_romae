import os

import torch
from roma.model import RoMAForClassification, RoMAForClassificationConfig
from roma.utils import load_from_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from elasticcv2_parq.dataset import ElasticcParquetDatasetwLabel
from elasticcv2_parq.config import ElasticcConfig
from sklearn.metrics import classification_report


def evaluate(args):
    #print("Implement evaluate")
    config = ElasticcConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on test set")
    
    # if args.pretrained_model is not None:
    #     print(f"Using {args.pretrained_model}")
    #     config.eval_checkpoint = args.pretrained_model

    
    model = load_from_checkpoint(args.eval_checkpoint, RoMAEForClassification,
                                 RoMAEForClassificationConfig).to(device).eval()
    all_preds = []
    all_labels = []
    with ( # Force mask_ratio to 0 ?
        ElasticcParquetDatasetwLabel(args.test_parquet, mask_ratio=0) as test_dataset,
    ):
        dataloader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            num_workers=os.cpu_count()-1,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {key: val.to(device) for key, val in batch.items()}
                logit, _ = model(**batch)
                preds = torch.argmax(torch.nn.functional.softmax(logit, dim=1), dim=1)
                all_labels.extend(list(batch["label"].cpu().numpy()))
                all_preds.extend(list(preds.cpu().numpy()))
    print(classification_report(
        all_labels,
        all_preds,
        ## Edit the following as I hard-coded it and it's ugly
        labels=list(range(32)),
        target_names= ['CART', 'CLAGN', 'Cepheid', 'EB', 'ILOT', 'KN_B19', 'KN_K17',
        'Mdwarf-flare', 'PISN', 'RRL', 'SLSN-I+host', 'SLSN-I_no_host',
        'SNII+HostXT_V19', 'SNII-NMF', 'SNII-Templates',
        'SNIIb+HostXT_V19', 'SNIIn+HostXT_V19', 'SNIIn-MOSFIT',
        'SNIa-91bg', 'SNIa-SALT3', 'SNIax', 'SNIb+HostXT_V19',
        'SNIb-Templates', 'SNIc+HostXT_V19', 'SNIc-Templates',
        'SNIcBL+HostXT_V19', 'TDE', 'd-Sct', 'dwarf-nova', 'uLens-Binary',
        'uLens-Single-GenLens', 'uLens-Single_PyLIMA'], #config.class_names,
        digits=4
    ))