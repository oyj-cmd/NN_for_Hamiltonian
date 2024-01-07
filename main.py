from typing import Dict, Tuple, List, Any
from torch import Tensor
import os
import torch
import numpy as np
import random
import time
import shutil
import json
import logging
import logging.config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from collections import namedtuple

from modelname.config import read_train_config, config_recorder
from modelname.graph import HeteroDataset
from modelname.old_0_model import Net
from modelname.graph import get_loss
from modelname.log import LOG_CONFIG
from modelname.config import write_default_train_config


# = To Be Developed = 
# 12/27 model.__repr__, __extr_repr__
# save_model -> load_model instead of build_model to avoid load checkpoint with different config.json
# logger.info instead of print


# (TODO Just For Code Developers) = Generate Configure JSON File =
root_dir: str = r"/home/muyj/Project/Project_1106_deephe3_example/modelname"
json_file =  os.path.join(root_dir, "config_dir", "config.json")
if os.path.exists(json_file):
    print(f"Configure JSON File has already been Generated in {json_file}.")
else:
    write_default_train_config(root_dir=root_dir) 


import warnings
warnings.filterwarnings("ignore")


# = Read Train Parameters = 
# TODO Variable json_file should come from users' parameters outside (ArgParse), named "config_file_path"
json_file = json_file
assert os.path.exists(json_file), f"Json file does not exist! \njson_file={json_file}"
config: config_recorder = read_train_config(json_file=json_file)


# = Initialize Logger =
LOG_CONFIG: Dict = json.loads(LOG_CONFIG)
LOG_CONFIG["handlers"]["file"]["filename"] = config.log_file
LOG_CONFIG["handlers"]["file"]["mode"] = 'w' #TODO whether to override existing log file ? 'w' for code developers
logging.config.dictConfig(config=LOG_CONFIG)
logger = logging.getLogger("main.py")
#TODO welcome page, need model name, cited information
logger.info(f"************************************************************")
logger.info(f"Welcome to The Model for Hamiltonians Prediction.")
logger.info(f"Cite Cite Cite Cite Cite Cite Cite")
logger.info(f"************************************************************")
# fill information of the last part
logger.info(f"# = Read Train Parameters = ")
logger.info(f"Reading Configure JSON File from {json_file}.")
# only save variables: config, logger
del root_dir, json_file, LOG_CONFIG



# = Initialize Seed = 
torch.manual_seed(config.seed)
torch.random.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# = Initialize Dataset =
logger.info(f"# = Initialize Dataset =") 
dataset: HeteroDataset = HeteroDataset(
    processed_dir=config.processed_dir, 
    graph_file=config.graph_file, 
    is_spin=config.is_spin, 
    default_dtype_torch=config.default_dtype_torch
    )
logger.debug(dataset)


# = Initialize Train Global Variables (1) = 
Ls_dict: Dict[int, List[int]] = dataset.Ls_dict # revert_blocks_dict: 用于把模型输出的直和形式哈密顿转化为张量积形式
default_dtype_torch: torch.dtype = dataset.default_dtype_torch # 如果图已生成，那么torch数据类型理应从图数据中读取
# torch.set_default_dtype(dataset.default_dtype_torch) 
sort_dict: List[int] = dataset.sort_dict # revert_blocks_dict: 用于把模型输出的简化Irreps排布方式按照原来的Irreps重新排序
# checkpoint 没有保存模型初始化信息，包括模型模块的输入输出Tensor形状等，如果读取checkpoint时模型初始化条件不同会出错
#(TODO 模型改变保存内容也要改) 用init_full_model_file保存模型初始化信息: 初次训练时的dataset.simple_sorted_block_irreps_dict
#TODO ？ Bug 不能直接保存原始模型，TypeError: write() argument must be str, not memoryview
init_full_model_file: str = os.path.join(config.save_dir, "init_full_model.pt") 


# = Initialize Model = 
logger.info(f"# = Initialize Model = ")
if os.path.exists(init_full_model_file) and os.path.exists(config.checkpoint_file):
    logger.info(f"Read the Initial Model Information from {init_full_model_file}.")
    model: Net = torch.load(init_full_model_file)
else:
    logger.info(f"Create an Initial Model.")
    model: Net = Net(block_irreps_dict=dataset.simple_sorted_block_irreps_dict)
    logger.info(f"Save the Initial Model into {init_full_model_file}.")
    torch.save(model, init_full_model_file)
logger.debug(f"The Initial Input For the Current Model: {model.init_model_input_for_debug}")
logger.debug(model)
model.to(config.device)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
# TODO change into logger.info(model) later, show parameters in __repr__ of model class
logger.info(f"Total Number of trainable Parameters in the Current Model: {params}") 
del params
# TODO why this code is necessary ?
model_parameters = filter(lambda p: p.requires_grad, model.parameters())


# = Initialize Optimizer =
if config.optimizer.lower()=='Adam'.lower():
    optimizer = torch.optim.Adam(model_parameters, **config.optimizer_params)
    # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    # optimizer_sgd = optim.SGD(model_parameters, lr=config.begin_lr)
    # TODO logger.info(f'Using optimizer Adam with initial lr={config.begin_lr}, betas={config.optimizer_adam_betas}')
else:
    raise NotImplementedError(f'Unsupported Optimizer: {config.optimizer}.')


# = Initialize Loss Function =
if config.criterion.lower()=='MSELoss'.lower():
    criterion = torch.nn.MSELoss()
    # TODO logger.info('Loss type: MSE over all matrix elements')
else:
    raise NotImplementedError(f'Unsupported Loss Function: {config.criterion}.')

# = Initialize Lr_scheduler = 
if config.scheduler.lower()=='ReduceLROnPlateau'.lower():
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        **config.scheduler_params
        )
else:
    raise NotImplementedError(f'Unsupported lr_scheduler: {config.scheduler}.')


# = Initialize Tensorboard Logger =
tb_writer = SummaryWriter(os.path.join(config.save_dir, 'tensorboard'))


# = Initialize Dataloader = 
logger.info(f'# = Initialize Dataloader = ')
indices = list(range(len(dataset)))
dataset_size = len(indices)
train_size = int(config.train_ratio * dataset_size)
val_size = int(config.val_ratio * dataset_size)
test_size = int(config.test_ratio * dataset_size)
np.random.shuffle(indices)
logger.info(f'Size of train set: {train_size}')
logger.info(f'Size of validation set: {val_size}')
logger.info(f'Size of test set: {test_size}')
train_loader = DataLoader(
    dataset, 
    batch_size=config.batch_size,
    shuffle=False, 
    sampler=SubsetRandomSampler(indices[:train_size])
    )
val_loader = DataLoader(
    dataset, 
    batch_size=config.batch_size,
    shuffle=False, 
    sampler=SubsetRandomSampler(indices[train_size: train_size + val_size])
    )
test_loader = DataLoader(
    dataset, 
    batch_size=config.batch_size,
    shuffle=False, 
    sampler=SubsetRandomSampler(indices[train_size + val_size : train_size + val_size + test_size])
    ) # (train_size + val_size + test_size) should <= dataset_size, or it will raise error here.
del dataset


# = Initialize Train Global Variables (2) = 
begin_time: float = time.time()
epoch_begin_time: float = time.time()
epoch: int = 1 # 训练总步数，包括checkpoint中的步数
best_val_loss: float = torch.inf


# = Read Checkpoint = 
logger.info(f'# = Read Checkpoint = ')
if os.path.exists(config.checkpoint_file):
    logger.info(f'Loading from checkpoint at {config.checkpoint_file}.')
    checkpoint = torch.load(
        config.checkpoint_file, 
        map_location='cpu'
        )
    # checkpoint: Dict, contains 'epoch', 'best_val_loss'
    #   'state_dict', 'optimizer_state_dict', 'scheduler_state_dict'
    epoch = int(checkpoint['epoch'])
    best_val_loss = float(checkpoint['best_val_loss'])
    model.load_state_dict(checkpoint['state_dict']) # model/optimizer/schedular is on GPU but load state from cpu
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    logger.info(f'Starting from epoch {checkpoint["epoch"]} with best validation loss {checkpoint["best_val_loss"]}.')
else:
    logger.info('No Checkpoint. Starting new training process.')
    pass

# = Train =
logger.info(f'# = Train = ')
try:
    while epoch <= config.max_num_epochs: 
        
        # = Train =
        train_losses: List = []
        for batch in train_loader:
            output: Dict = model(batch.to(device=config.device))
            loss = get_loss(
                    criterion=criterion,
                    y_dict=batch.y_dict,
                    out_dict=output,
                    sort_dict=sort_dict, 
                    Ls_dict=Ls_dict, 
                    is_spin=config.is_spin,
                    default_dtype_torch=config.default_dtype_torch, 
                    device=config.device
                )
            # backward propagation
            optimizer.zero_grad()
            loss.backward() # calculation graph has been destroyed
            optimizer.step()
            # record train loss
            train_losses.append(loss.item())
        
        
        # = Validation =
        with torch.no_grad():
            val_losses: List = []
            for batch in val_loader:
                output: Dict = model(batch.to(device=config.device))
                loss = get_loss(
                    criterion=criterion,
                    y_dict=batch.y_dict,
                    out_dict=output,
                    sort_dict=sort_dict, 
                    Ls_dict=Ls_dict, 
                    is_spin=config.is_spin,
                    default_dtype_torch=config.default_dtype_torch, 
                    device=config.device
                )
                # record validation loss
                val_losses.append(loss.item())


        # = Test =   
        with torch.no_grad():
            test_losses: List = []
            for batch in test_loader:
                output: Dict = model(batch.to(device=config.device))
                loss = get_loss(
                    criterion=criterion,
                    y_dict=batch.y_dict,
                    out_dict=output,
                    sort_dict=sort_dict, 
                    Ls_dict=Ls_dict, 
                    is_spin=config.is_spin,
                    default_dtype_torch=config.default_dtype_torch, 
                    device=config.device
                )
                test_losses.append(loss.item())


        # = Record Loss =
        learning_rate = optimizer.param_groups[0]['lr']
        train_info_recorder = namedtuple('train_info_recorder', 
        ['epoch', 
         'lr', 
        'total_time', 'epoch_time', 
        'train_loss', 'val_loss', 'test_loss'])
        train_info_recorder = train_info_recorder(
            epoch=epoch, 
            lr=learning_rate,
            total_time=time.time()-begin_time,
            epoch_time=time.time()-epoch_begin_time,
            train_loss=sum(train_losses)/len(train_losses),
            val_loss=sum(val_losses)/len(val_losses),
            test_loss=sum(test_losses)/len(test_losses)
            )
        

        # = Print Loss =
        time_r = round(train_info_recorder.total_time)
        d, h, m, s = time_r//86400, time_r%86400//3600, time_r%3600//60, time_r%60
        out_info = (f'Epoch #{train_info_recorder.epoch:<5d} | '
                    f'Time: {d:02d}d {h:02d}h {m:02d}m | '
                    f'LR: {train_info_recorder.lr:.2e} | '
                    f'Epoch time: {train_info_recorder.epoch_time:6.2f} | '
                    f'Train loss: {train_info_recorder.train_loss:.2e} | ' # :11.8f
                    f'Val loss: {train_info_recorder.val_loss:.2e} | '
                    f'Test loss: {train_info_recorder.test_loss:.2e} '
                    )
        logger.info(out_info)
        

        # = Record in Tensorboard =
        tb_writer.add_scalar('Learning rate', train_info_recorder.lr, global_step=train_info_recorder.epoch)
        tb_writer.add_scalars('Loss', {'Train loss': train_info_recorder.train_loss}, global_step=train_info_recorder.epoch)
        tb_writer.add_scalars('Loss', {'Validation loss': train_info_recorder.val_loss}, global_step=train_info_recorder.epoch)
        tb_writer.add_scalars('Loss', {'Test loss': train_info_recorder.test_loss}, global_step=train_info_recorder.epoch)
            
        '''
        # = Write Report =
        #TODO  只记录最优模型那一训练步骤的信息
        if train_info_recorder.val_loss < best_val_loss:
            file = open(os.path.join(config.save_dir, 'train_report.txt'), 'w')
            logger.info(f'Best model:', file=file)
            logger.info(out_info, end='\n', file=file)
            file.close()
        '''
        



        # = Save Checkpoint and Update Train Global Variables =
        is_best: bool = False
        if train_info_recorder.val_loss < best_val_loss:
            best_val_loss = train_info_recorder.val_loss
            is_best = True

        checkpoint: Dict[str, Any] = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(
            checkpoint, 
            os.path.join(config.checkpoint_file)
            )

        if is_best and epoch != 1:
            shutil.copyfile(
                os.path.join(config.checkpoint_file),
                os.path.join(config.save_dir, 'best_model.pt')
            )
            
        scheduler.step(train_info_recorder.val_loss)
        epoch_begin_time = time.time()
        epoch += 1 #epoch = scheduler.next_epoch

    logger.info(f"MAX Epochs Number {config.max_num_epochs} has been Reached.")
    logger.info("Normal Termination of Training Process.")   
except KeyboardInterrupt:
    logger.info("KeyboardInterrupt")
    logger.info("Deliberate interruption of Training Process.")  

logger.info(f"# = Summary of Memory/Time Usage = ")
time_r = round(time.time() - begin_time)
d, h, m, s = time_r//86400, time_r%86400//3600, time_r%3600//60, time_r%60
logger.info(f"Total Time: {d:02d}d {h:02d}h {m:02d}m.")  
#TODO Memory Usage and Time Usage Summary
#TODO Analyze the Model



