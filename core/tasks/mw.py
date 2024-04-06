import pdb

import hydra.utils

from .base_task import  BaseTask
from core.data.vision_dataset import VisionData
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import datetime
from core.utils import *
import glob
import omegaconf
import json
from eval_policy import eval_policy, build_env_policy


class CFTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(CFTask, self).__init__(config, **kwargs)

    # override the abstract method in base_task.py
    def set_param_data(self):
        param_data = PData(self.cfg.param)
        return param_data
    
    def build_env_policy(self):
        env, policy = build_env_policy()
        self.env = env
        self.policy = policy

    def test_g_model(self, param): 
        avg_success, avg_reward, avg_step_length = eval_policy(param, self.env, self.policy)
        return  avg_success, avg_reward, avg_step_length




