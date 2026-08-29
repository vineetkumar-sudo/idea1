import torch
from pcdet.datasets.cadc.cadc_dataset import CadcDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pathlib import Path
import os

def test_loader():
    config_path = Path("OpenPCDet/tools/cfgs/dataset_configs/cadc_dataset.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    cfg_from_yaml_file(config_path, cfg)
    
    dataset_cfg = cfg 
    dataset_cfg.DATA_PATH = os.path.abspath("data/cadc")
    
    # Disable augmentation and GT sampling for the smoke test
    dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['gt_sampling', 'random_world_flip', 'random_world_rotation', 'random_world_scaling']
    
    print(f"Data Path: {dataset_cfg.DATA_PATH}")
    
    print("Initializing CADC Dataset (Eval Mode)...")
    try:
        # Set training=False to avoid looking for training-specific .pkl files
        cadc_dataset = CadcDataset(
            dataset_cfg=dataset_cfg,
            class_names=['Car'],
            training=False, 
            root_path=Path(dataset_cfg.DATA_PATH)
        )
        
        # If it still asks for cadc_infos_val.pkl, we may need to 
        # manually mock the sample_id_list for this test.
        if not hasattr(cadc_dataset, 'sample_id_list') or len(cadc_dataset.sample_id_list) == 0:
            print("Manually setting sample ID for smoke test...")
            cadc_dataset.sample_id_list = ['2019_02_27/0002/000000'] # Adjust to your actual file name
        
        print(f"Dataset length: {len(cadc_dataset)}")
        
        if len(cadc_dataset) > 0:
            # Try loading the first frame
            data_dict = cadc_dataset[0]
            print("✅ Successfully loaded first frame!")
            print(f"Points shape: {data_dict['points'].shape}")
            if 'gt_boxes' in data_dict:
                print(f"GT Boxes shape: {data_dict['gt_boxes'].shape}")
        else:
            print("❌ Dataset is empty.")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("\nTip: If it complains about 'cadc_infos_val.pkl', we need to run the data prep script.")

if __name__ == "__main__":
    test_loader()