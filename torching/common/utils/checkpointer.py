import os
import torch
import glob
import re
import time
from datetime import datetime
from datetime import timedelta
import logging


class AbcCheckpointer:
    def save(self, state_dict, ignore_period=True):
        raise NotImplementedError

    def load(self, model):
        raise NotImplementedError
    

class CheckPointer(AbcCheckpointer):
    def __init__(self,
                 checkpoint_dir,
                 max_checkpoints=10,
                 checkpoint_period=None,
                 name=None,
                 logger=None):
        self.name = name
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception('The checkpoints directory is invalid')

        self.max_checkpoints = max_checkpoints

        if isinstance(checkpoint_period, timedelta):
            self.checkpoint_period = checkpoint_period
        elif isinstance(checkpoint_period, (float, int)):
            self.checkpoint_period = timedelta(seconds=int(checkpoint_period))
        else:
            self.checkpoint_period = None

        self._due_time = datetime.now() if self.checkpoint_period else None

        logger_name = 'Checkpointer.{}'.format(self.name) if self.name else 'Checkpointer'
        self.logger = logger or logging.getLogger(logger_name)

    def save(self, state_dict, ignore_period=True):
        if self._due_time:
            if datetime.now() > self._due_time:
                self._due_time = datetime.now() + self.checkpoint_period
            elif not ignore_period:
                return False
                
        try:
            torch.save(state_dict, self._make_new_checkpoint_filepath())
            return True
        except Exception as e:
            self.logger.warning('Failed to save as checkpoint: {}'.format(e))
            return False

    def load(self, model):
        filepath = self._find_latest_checkpoint_file_in_the_dir()
        if not filepath:
            self.logger.warning(f'There is no checkpoints in the `{self.checkpoint_dir}`')
            return False

        model.load_state_dict(torch.load(filepath))
        return True

    def _make_new_checkpoint_filepath(self):
        name = self.name or 'awesome_model'
        filaname =  '{}.pth.{}'.format(name, int(time.time() * 1000))
        filepath = os.path.join(self.checkpoint_dir, filaname)
        return filepath

    def _find_latest_checkpoint_file_in_the_dir(self):
        filepaths = glob.glob(os.path.join(self.checkpoint_dir, '*.pth.*'))

        if not any(filepaths):
            return
        
        unique_names = set()
        timestamps = []
        for filepath in filepaths:
            filename, _ = os.path.split(filepath)
            name, _, suffix = filename.split('.')
            unique_names.add(name)
            timestamps.append(suffix)
        timestamps.sort()

        assert len(unique_names) == 1, Exception('')
        the_lastest = os.path.join(self.checkpoint_dir, '{}.pth.{}'.format(unique_names.pop(), suffix[-1]))
        return the_lastest