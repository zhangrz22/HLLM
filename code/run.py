# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from cProfile import run
from logging import getLogger
import torch
import json
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist

import os
import numpy as np
import argparse
import torch.distributed as dist
import torch

def run_loop(local_rank, config_file=None, saved=True, extra_args=[]):

    # configurations initialization
    config = Config(config_file_list=config_file)

    device = torch.device("cuda", local_rank)
    config['device'] = device
    config.parse_extra_args(extra_args)

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    if 'text_path' in config:
        config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
        logger.info(f"Update text_path to {config['text_path']}")

    # get model and data
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    print(f"{len(train_loader) = }")

    model = get_model(config['model'])(config, dataload)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    world_size = torch.distributed.get_world_size()
    trainer = Trainer(config, model)

    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)
    logger.info(dataload)
    logger.info(model)

    if config['val_only']:
        ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        logger.info(f'Eval only model load from {ckpt_path}')
        msg = trainer.model.load_state_dict(ckpt, False)
        logger.info(f'{msg.unexpected_keys = }')
        logger.info(f'{msg.missing_keys = }')
        test_result = trainer.evaluate(test_loader, load_best_model=False, show_progress=config['show_progress'], init_model=True)
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    else:
        # training process
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Trianing Ended' + set_color('best valid ', 'yellow') + f': {best_valid_result}')

        # model evaluation
        test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    config_file = args.config_file

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
