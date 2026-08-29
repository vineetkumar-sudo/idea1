"""Gate 200 smoke test: load a CADC frame through the full OpenPCDet pipeline."""
import pickle
import yaml
import numpy as np
from pathlib import Path
from easydict import EasyDict

from pcdet.datasets.cadc.cadc_dataset import CadcDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'cadc'
CFG = ROOT / 'OpenPCDet' / 'tools' / 'cfgs' / 'dataset_configs' / 'cadc_dataset.yaml'

ALL_CLASSES = ['Car', 'Pedestrian', 'Pickup_Truck']


def classes_for(training):
    """gt_sampling requires every class name to exist in cadc_dbinfos_train.pkl.
    The smoke-test subset (2019_02_27/0002) has no Pedestrian instances, so in
    training mode restrict to the classes the gt database actually contains."""
    if not training:
        return ALL_CLASSES
    db = pickle.load(open(DATA / 'cadc_dbinfos_train.pkl', 'rb'))
    return [c for c in ALL_CLASSES if c in db]


def report(ds, classes, mode):
    print(f'\n=== {mode}  (classes: {classes}) ===')
    print(f'Dataset length   : {len(ds)} frames')
    d = ds[0]
    pts = d['points']
    print(f"frame id         : {'/'.join(d['sample_idx'])}")
    print(f"points           : {pts.shape}  "
          f"x[{pts[:, 0].min():.1f},{pts[:, 0].max():.1f}] "
          f"z[{pts[:, 2].min():.1f},{pts[:, 2].max():.1f}] "
          f"intensity[{pts[:, 3].min():.2f},{pts[:, 3].max():.2f}]")
    print(f"voxels           : {d['voxels'].shape}")
    print(f"voxel_coords     : {d['voxel_coords'].shape}")
    print(f"voxel_num_points : {d['voxel_num_points'].shape}")
    gt = d['gt_boxes']
    names = [classes[int(c) - 1] for c in gt[:, 7]]
    uniq, cnt = np.unique(names, return_counts=True)
    print(f"gt_boxes         : {gt.shape}  {dict(zip(uniq, cnt.tolist()))}")


if __name__ == '__main__':
    for training in (False, True):
        classes = classes_for(training)
        cfg = EasyDict(yaml.safe_load(open(CFG)))
        cfg.DATA_PATH = str(DATA)
        ds = CadcDataset(dataset_cfg=cfg, class_names=classes,
                         training=training, root_path=DATA)
        report(ds, classes, 'TRAIN' if training else 'VAL')
