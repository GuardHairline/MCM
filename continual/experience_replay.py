import argparse
import random
import logging
from typing import List, Dict

import torch
from torch import nn
from torch.optim import AdamW

from datasets.get_dataset import get_dataset  # 这里引用你现有的 get_dataset 函数
from utils.logging import setup_logger

logger = logging.getLogger(__name__)


class ExperienceReplayMemory:

    def __init__(self):
        '''
        Initializes ER memory with empty memory buffer dict
        '''
        self.memory_buffers = {}

    def add_task_memory_buffer(self, 
                               args: argparse.Namespace, 
                               task_name: str,  # 使用任务名称作为标识符
                               task_config: Dict,
                               memory_percentage: float, 
                               sampling_strategy: str):
        '''
        Creates a memory buffer for new task
        '''

        task_buffer = TaskMemoryBuffer(args, task_name, task_config, memory_percentage, sampling_strategy)
        self.memory_buffers[task_name] = task_buffer

    def do_replay(self) -> bool:
        '''
        Return true if there are any tasks in the memory to do replay on, else False
        '''
        return True if len(self.memory_buffers) > 0 else False

    def sample_replay_task(self) -> str:
        '''
        Samples a previous task at random
        '''
        previous_tasks = list(self.memory_buffers.keys())
        #ran_int = random.randint(0,1)
        #if ran_int < 0.3 and 'cocoqa' in previous_tasks:
        #    sampled_previous_task = 'cocoqa'
        #elif ran_int < 0.7:
        #    sampled_previous_task = 'nlvr2'
        #else:
        sampled_previous_task = random.choice(previous_tasks)
        return sampled_previous_task

    def create_optimizer(self, model, task_name):

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        lr = 1e-4
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
        return optimizer

    def run_replay_step(self, task_name: str, model) -> torch.Tensor:
        '''
        Performs a single training step on previous task, by sampling a batch from task bugger
        '''
        task_buffer = self.memory_buffers[task_name]
        task_config = task_buffer.task_config

        optimizer = self.create_optimizer(model,task_name)
        replay_batch = task_buffer.sample_replay_batch()

        loss = model(replay_batch)  # Replace with your model's actual forward pass and loss computation
        loss.backward()
        optimizer.step()

        logger.info(f"{task_name} replay step: loss = {loss.item()}")
        return loss

class TaskMemoryBuffer:

    '''
    Buffer of training examples that can be used for replay steps
    '''
    def __init__(self, 
                 args: argparse.Namespace, 
                 task_name: str, 
                 task_config: Dict, 
                 memory_percentage: float,
                 sampling_strategy: str):

        '''
        Creates a memory buffer for new task, which samples a small percentage of training data for experience replay
        '''

        self.task_name = task_name
        self.task_config = task_config

        self.dataset = get_dataset(task_name, "train", args)  # 使用你的 get_dataset 来加载任务数据集
        self.batch_collate_fn = self.dataset.collate_fn  # Assuming the dataset has a collate_fn method
        self.batch_size = args.batch_size

        self.memory_percentage = memory_percentage                      # Percent of training samples to store in memory
        assert self.memory_percentage < 1.0
        self.memory_size = int(memory_percentage*len(self.dataset))     # Number of training samples that are stored in memory
        self.sampling_strategy = sampling_strategy
        assert sampling_strategy in ['random']                      # Only random sampling for memory buffer implemented so far

        if self.sampling_strategy == 'random':
            train_idxs = list(range(len(self.dataset)))
            self.memory_idxs = random.sample(train_idxs, self.memory_size)

        # elif self.sampling_strategy == 'random-balanced':
        #     raise NotImplementedError("Label-balanced sampling of replay memory is not yet implemented!")

        logger.info("Created {} replay memory buffer, with {} samples in the memory".format(self.task_name, len(self.memory_idxs)))

    def __len__(self):
        return len(self.memory_idxs)

    def sample_replay_batch(self) -> Dict:

        sampled_instances = random.sample(self.memory_idxs, self.batch_size)
        batch = self.batch_collate_fn([self.dataset[i] for i in sampled_instances])
        return batch
