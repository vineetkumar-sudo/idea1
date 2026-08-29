"""Generate CADC info/db pickles for the sequences listed in data/cadc/ImageSets/.

Bypasses cadc_dataset.py's __main__ block, which hardcodes data_path to
OpenPCDet/data/cadc and reads DATA_PATH: '/root/cadc' from the yaml.
"""
import yaml
from pathlib import Path
from easydict import EasyDict

from pcdet.datasets.cadc.cadc_dataset import create_cadc_infos

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'cadc'
CFG = ROOT / 'OpenPCDet' / 'tools' / 'cfgs' / 'dataset_configs' / 'cadc_dataset.yaml'

if __name__ == '__main__':
    dataset_cfg = EasyDict(yaml.safe_load(open(CFG)))
    dataset_cfg.DATA_PATH = str(DATA)          # overrides '/root/cadc'

    create_cadc_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Pickup_Truck'],
        data_path=DATA,                        # -> imagesets_path
        save_path=DATA,                        # -> where the .pkl files land
        workers=4,
    )
