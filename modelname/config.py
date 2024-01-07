from typing import Dict
from collections import namedtuple
import os
import json
import torch

from .graph import to_dtype_default_torch_dtype

config_recorder = namedtuple(
    typename='config_recorder',
    field_names=[
        'processed_dir', 'is_spin', 'default_dtype_torch',
        'graph_file', 'save_dir', 'checkpoint_file', 'log_file',
        'batch_size',
        'optimizer', 'optimizer_params',
        'criterion',
        'scheduler', 'scheduler_params',
        'max_num_epochs', 
        'device',
        'train_ratio', 'val_ratio', 'test_ratio',
        'seed'
    ]
)

def read_train_config(json_file: str) -> config_recorder:
    with open(json_file, 'r') as f:
        dct: Dict = json.load(f)
        if isinstance(dct['default_dtype_torch'], str):
            dct['default_dtype_torch'] = to_dtype_default_torch_dtype(dct['default_dtype_torch'], dtype=torch.dtype)
        if dct['optimizer'].lower()=='Adam'.lower() and dct['optimizer_params']['betas'] is not None:
            dct['optimizer_params']['betas'] = tuple(dct['optimizer_params']['betas'])
        config: config_recorder = config_recorder(**dct)
    return config

# For Code Developers
def write_train_config(json_file: str, config: config_recorder, verbose: bool=False):
    # Use JSON instead of configparser !
    # It will raise "TypeError: option values must be strings" without allow_no_value=True
    # config_parser = configparser.ConfigParser(allow_no_value=True) 
    # dct = dict({'Train Parameters': config._asdict()})
    dct: Dict = config._asdict()
    if isinstance(dct['default_dtype_torch'], torch.dtype):
        dct['default_dtype_torch'] = str(dct['default_dtype_torch'])
    if dct['optimizer'].lower()=='Adam'.lower() and dct['optimizer_params']['betas'] is not None:
        dct['optimizer_params']['betas'] = list(dct['optimizer_params']['betas'])
    print(f"(For Code Developers Only) Write configurations to {json_file}.")
    if verbose == True:
        print(f"Dictionary of configuration: \n{json.dumps(dct, indent=4)}")
    with open(json_file, 'w') as f:
        json.dump(dct, f, indent=4)

def get_default_train_config(root_dir: str) -> config_recorder:
    default_train_config = config_recorder(
            # = Dataset Parameters =
            # processed_dir=r"/home/muyj/Project/Project_1106_deephe3_example/bismuth_workdir/3_openmx_processed", 
            processed_dir=os.path.join(
                root_dir, 
                "data_dir"),                    # 如果图已生成，那么理应不需提供量子化学软件数据 (None)
            is_spin=True,                       # 如果图已生成，那么理应不需提供 (None)
            default_dtype_torch=str(torch.float32),  # 如果图已生成，那么理应从图数据中读取 (None)
            # = Input/Output Path Parameters =
            graph_file=os.path.join(root_dir, "graph_dir", "graph.pt"),
            save_dir=os.path.join(root_dir, "save_dir"), 
            checkpoint_file=os.path.join(root_dir, "save_dir", "model.pt"), 
            log_file=os.path.join(root_dir, "save_dir", "train_report.log"),
            # = Train Parameters =
            # 批量大小
            batch_size=2, 
            # 优化器
            optimizer='Adam',                   # 优化器名称不分辨大小写
            optimizer_params={
                'lr': 0.01, 
                'betas': (0.9, 0.99)
                }, 
            # 损失函数
            criterion='MSELoss',                # 损失函数名称不分辨大小写
            # 学习率调节器
            scheduler='ReduceLROnPlateau',      # 学习率调节器名称不分辨大小写
            scheduler_params={
                'factor': 0.5, 
                'cooldown': 40, 
                'patience': 120, 
                'threshold': 0.05, 
                'verbose': True
                }, 
            # 最大训练步数
            max_num_epochs=10000, 
            # 设备(cpu/gpu/gpu groups TODO)
            device='cuda:0',
            # 训练集，验证集，测试集占比
            train_ratio=0.6, 
            val_ratio=0.2, 
            test_ratio=0.2, 
            # = Seed Parameters = 
            seed=1,
        )
    
    return default_train_config

def write_default_train_config(root_dir: str, verbose: bool=False):
    """Easy configure JSON file generator for code developers.
    
    Note that JSON file path: `/${root_dir}/config_dir/config.json`. """
    json_file =  os.path.join(root_dir, "config_dir", "config.json")
    default_train_config: config_recorder = get_default_train_config(root_dir)
    write_train_config(json_file=json_file, config=default_train_config, verbose=verbose)
    return None

    
    