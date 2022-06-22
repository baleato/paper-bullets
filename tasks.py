import math
from collections import Counter

import numpy as np
import logging

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import BCEWithLogitsLoss

from models import SLClassifier


class Task(object):
    r"""Base class for every task."""
    NAME = 'TASK_NAME'

    def __init__(self, cls_dim=768, pos_weight=True):
        self.num_classes = None
        self.dataset_class = None

    def get_dataloader(self, split, tokenizer, batch_size=32, random_state=1, shuffle=True, drop_last=True):
        """
        Returns an iterable over the single
        Args:
            split: train/dev/test
        Returns:
            Iterable for the specified split
        """
        raise NotImplementedError

    def get_classifier(self):
        return self.classifier

    def get_loss(self, predictions, labels):
        return self.criterion(predictions, labels)

    # TODO: revise metric
    def score(self, y_true, y_pred):
        """
        Returns: precision, recall and F-score
        """
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        return precision, recall, fscore, support

    # TODO: remove
    def calculate_accuracy(self, predictions, labels):
        new_predictions = predictions.argmax(dim=1, keepdim=False)
        bin_labels = new_predictions == labels
        correct = bin_labels.sum().float().item()
        return correct / len(labels)

    def get_name(self):
        return self.NAME

    def get_num_classes(self):
        return self.num_classes

    def describe(self):
        print('No description provided for task {}'.format(self.get_name()))


class TaskSamplerIter(object):
    """Iterator class used by TaskSampler.
    Parameters:
        - dataloaders
        - task_sampling_ratio ("equal", "sqrt" or Array<float>)
            - equal: keeps sampling tasks sequentially
            - sqrt: applies a task sampling ratio based on the square root of the size of each dataset
            - Array<float>: ratios. This values will be transform to probabilities.
              E.g. [0.25, 0.75] and [1, 3] are equivalent.
        - epoch_factor (Array<float>): A factor applied to the task sampling ratio each time the iterator for the task
          is exhausted. This can be use to decay the sampling ratio of a particular task overtime.
        - alignment_strategy ("prop", "max", "min" (default)). Used to calculate how many batches fit into the current
          epoch.
            - "prop" (proportional): the size of the epoch is calculated as the dot product between the task sampling
              probability and the number of batches for that task.
            - "min1": limits the total number of batches per task in the epoch to the number of batches of the smallest
              task. Sampling ratios are not taking into consideration.
            - "max1": limits the total number of batches per task in the epoch to the number of batches of the smallest
              task. Sampling ratios are not taking into consideration.
    """
    def __init__(self, dataloaders, task_sampling_ratio='equal', epoch_factor=None, alignment_strategy="min1"):
        self.original_dataloaders = dataloaders
        self.task_iters = [iter(dl) for dl in dataloaders]
        self.task_sampling_ratio = task_sampling_ratio
        self.epoch_factor = np.array(epoch_factor if epoch_factor else [1] * len(dataloaders))
        self.alignment_strategy = alignment_strategy
        if task_sampling_ratio == 'sqrt':
            # Using the square root of the dataset size is a strategy that yields good results.
            # Additionally, we divide by the number of times the same dataset is used in
            # different tasks. This aims to attenuate bias towards the data distribution of
            # a particular dataset.
            task_ratio = [math.sqrt(len(dl)) for dl in self.original_dataloaders]
        elif task_sampling_ratio == 'equal':
            task_ratio = [1] * len(self.task_iters)
        else:
            task_ratio = task_sampling_ratio
        self.task_ratio = task_ratio
        self.set_probabilities(task_ratio)
        # TODO: think about removing 'equal' and use only probabilities (ponder that it was called 'sequential' before)
        assert not epoch_factor or task_sampling_ratio != 'equal',\
            'Equal task sampling is not supported in combination with epoch_factor'
        self.task_index = 0
        self.batch_idx = 0
        self._calculate_num_total_batches()

    def get_task_index(self):
        return self.task_index

    def sample_next_task(self):
        if self.task_sampling_ratio == 'equal':
            return (self.task_index + 1) % len(self.task_iters) if self.batch_idx != 0 else 0
        else:
            return np.random.choice(len(self.task_iters), p=self.task_probs)

    def set_probabilities(self, task_ratio):
        # Restart task probabilities
        self.task_probs = [tr / sum(task_ratio) for tr in task_ratio]

    def _recalculate_sampling_probabilities(self):
        new_ratios = np.multiply(np.array(self.task_probs), self.epoch_factor)
        self.task_probs = [ratio / sum(new_ratios) for ratio in new_ratios]

    def _calculate_num_total_batches(self):
        task_lengths = [len(task_iter) for task_iter in self.task_iters]
        if self.alignment_strategy == "min1":
            self.num_total_batches = min(task_lengths)
        elif self.alignment_strategy == "max1":
            self.num_total_batches = max(task_lengths)
        elif self.alignment_strategy == "prop":
            self.num_total_batches = int(np.dot(np.array(task_lengths), np.array(self.task_probs)))
        else:
            assert False, f'Alignment strategy [{self.alignment_strategy}] not yet implemented!'

    def __iter__(self):
        self._calculate_num_total_batches()
        return self

    def __next__(self):
        if self.task_iters:
            if self.batch_idx == self.num_total_batches:
                # In the sampling procedure it could happen that a task finishes its examples
                # earlier than the rest of the tasks. If this happens we restart the iterator
                # for that task so we can keep delivering batches. Here, we artificially raise
                # the StopIteration exception when the number of batches returned matches the
                # total number of batches previously calculated.
                self.batch_idx = 0
                self._recalculate_sampling_probabilities()
                raise StopIteration

            task_index = self.sample_next_task()
            task_iter = self.task_iters[task_index]

            try:
                batch = next(task_iter)
            except StopIteration:
                # Note that depending on how next it's implemented it could also
                # return an empty list instead of raising StopIteration

                # if iterator is empty initialize new iterator from original dataloader
                task_iter = iter(self.original_dataloaders[task_index])
                self.task_iters[task_index] = task_iter
                batch = next(task_iter)

            self.task_index = task_index
            self.batch_idx += 1
            return batch
        else:
            raise StopIteration

    def __len__(self):
        return self.num_total_batches


class TaskSampler(Task):
    r"""This sampler is implemented as a task.
        task_all = TaskSampler([
                            Task_A(),
                            Task_B(),
                            Task_C(),
                        ])
        train_iter = task_all.get_dataloader('train')
        for batch in train_iter:
            ...
    """

    # Improvements on task sampler:
    #   - [X] Allow to specify sampling factors per task. For instance: [1, 2, 0.5, 0.5]
    #     will sample task 1 (25%), task 2 (50%) and task 3 and 4 (12.5%) each.
    #   - [X] Mind imbalance data (-> sample freq. sqrt of dataset length)
    def __init__(self, tasks, task_sampling_ratio='equal', epoch_factor=None, alignment_strategy="min1"):
        assert len(tasks) > 0
        self.tasks = tasks
        self.custom_task_ratio = task_sampling_ratio
        self.epoch_factor = epoch_factor
        self.alignment_strategy = alignment_strategy

    def get_dataloader(self, split, tokenizer, batch_size=32, random_state=1, shuffle=True, drop_last=True):
        task_iters = [
            task.get_dataloader(split, tokenizer, batch_size, random_state=random_state, shuffle=shuffle, drop_last=drop_last)
            for task in self.tasks
        ]
        self._task_sampler_iter = TaskSamplerIter(task_iters, self.custom_task_ratio, self.epoch_factor, self.alignment_strategy)
        return self._task_sampler_iter

    def _get_current_tasks(self):
        task_index = self._task_sampler_iter.get_task_index()
        return self.tasks[task_index]

    def get_task(self, task_index):
        return self.tasks[task_index]

    def get_classifier(self):
        return self._get_current_tasks.get_classifier()

    def get_loss(self, predictions, labels):
        return self._get_current_tasks().get_loss(predictions, labels)

    def calculate_accuracy(self, predictions, labels):
        return self._get_current_tasks().calculate_accuracy(predictions, labels)

    def get_name(self):
        return self._get_current_tasks().get_name()

    def get_task_index(self):
        return self._task_sampler_iter.get_task_index()

    def get_num_classes(self):
        return self._get_current_tasks().num_classes

# FIXME: retrieve global device instead
def _get_pytorch_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

from data import PTC, MPTC, fn_tokenizer, PTCLoadedLanguage, PTCNameCallingLabeling, MPTCLoadedLanguage, \
    MPTCNameCallingLabeling
from torch.utils.data import DataLoader
class PropagandaFLCTask(Task):
    """
    Propaganda Fragment Level Classification Task
    """
    NAME = 'Propaganda'
    DATASET_CLASS = PTC

    def __init__(self, cls_dim=768, pos_weight=True):
        super().__init__(cls_dim)
        self.num_classes = len(self.DATASET_CLASS.LABELS)
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = BCEWithLogitsLoss(
            pos_weight=self._calculate_pos_weights().to(_get_pytorch_device()) if pos_weight else None
        )
        # TODO: remove instance property dataset_class
        self.dataset_class = self.DATASET_CLASS
        # Lazy initialization
        self.dataset = None
        self.dataloader = None

    def get_dataloader(self, split, tokenizer, batch_size=32, random_state=1, shuffle=True, drop_last=True):
        self.dataset = self.DATASET_CLASS(split)
        # TODO: move collate lambda to data.py
        collate_fn = lambda *params: fn_tokenizer(*params, tokenizer=tokenizer, num_bin_classes=len(self.DATASET_CLASS.LABELS))
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=drop_last,
                                     shuffle=shuffle)
        return self.dataloader

    def describe(self):
        print('Propaganda Fragment Level Classification Task')

    def _calculate_pos_weights(self):
        df = self.DATASET_CLASS('train').df
        counter = Counter()
        total_words = 0
        for example in df.itertuples():
            total_words += len(example.sentence.split(' '))
            for technique_id, (start, end) in example.spans:
                technique_name = self.DATASET_CLASS.LABELS[technique_id]
                n_words = len(example.sentence[start:end].split(' '))
                counter.update([technique_name] * n_words)
        counts = [counter.get(technique_name) for technique_name in self.DATASET_CLASS.LABELS]
        pos_freq = torch.tensor(counts)
        neg_freq = total_words - pos_freq
        pos_weights = neg_freq / pos_freq
        return pos_weights


class PTCLoadedLanguageTask(PropagandaFLCTask):
    NAME = 'Propaganda_LoadedLanguage'
    DATASET_CLASS = PTCLoadedLanguage


class PTCNameCallingLabelingTask(PropagandaFLCTask):
    NAME = 'Propaganda_NameCalling'
    DATASET_CLASS = PTCNameCallingLabeling


class MemesTask(Task):
    """
    Meme Propaganda Classification Task
    """
    NAME = 'Memes'
    DATASET_CLASS = MPTC

    def __init__(self, cls_dim=768, pos_weight=True):
        super().__init__(cls_dim)
        self.num_classes = len(self.DATASET_CLASS.LABELS)
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = BCEWithLogitsLoss(
            pos_weight=self._calculate_pos_weights().to(_get_pytorch_device()) if pos_weight else None
        )
        self.dataset_class = self.DATASET_CLASS

    def get_dataloader(self, split, tokenizer, batch_size=32, random_state=1, shuffle=True, drop_last=True):
        self.dataset = self.DATASET_CLASS(split)
        collate_fn = lambda *params: fn_tokenizer(*params, tokenizer=tokenizer, num_bin_classes=len(self.DATASET_CLASS.LABELS))
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=drop_last,
                                     shuffle=shuffle)
        return self.dataloader

    def describe(self):
        print('Meme Propaganda Classification Task')

    def _calculate_pos_weights(self):
        df = self.DATASET_CLASS('train').df
        counter = Counter()
        total_words = 0
        for example in df.itertuples():
            total_words += len(example.text.split(' '))
            for label in example.labels:
                technique_name = label.get('technique')
                start = label.get('start')
                end = label.get('end')
                n_words = len(example.text[start:end].split(' '))
                counter.update([technique_name] * n_words)
        counts = [counter.get(technique_name) for technique_name in self.DATASET_CLASS.LABELS]
        pos_freq = torch.tensor(counts)
        pos_weights = (total_words - pos_freq) / pos_freq
        return pos_weights


class MemesLoadedLanguageTask(MemesTask):
    NAME = 'Memes_LoadedLanguage'
    DATASET_CLASS = MPTCLoadedLanguage


class MemesNameCallingLabelingTask(MemesTask):
    NAME = 'Memes_NameCalling'
    DATASET_CLASS = MPTCNameCallingLabeling


from data import VUAMetaphorDataset, VUAMetaphorSharedTaskAllPOSDataset, VUAMetaphorSharedTaskVerbDataset
class VUAMetaphorTask(Task):
    """
    Metaphor Classification Task
    """
    NAME = 'VUAMetaphor'
    DATASET_CLASS = VUAMetaphorDataset
    def __init__(self, cls_dim=768, pos_weight=True):
        super().__init__(cls_dim)
        self.num_classes = len(VUAMetaphorDataset.LABELS)
        self.classifier = SLClassifier(input_dim=cls_dim, target_dim=self.num_classes)
        self.criterion = BCEWithLogitsLoss(
            pos_weight=self._calculate_pos_weights().to(_get_pytorch_device()) if pos_weight else None
        )
        self.dataset_class = self.DATASET_CLASS
        # Lazy initialization
        self.dataset = None
        self.dataloader = None

    def get_dataloader(self, split, tokenizer, batch_size=32, random_state=1, shuffle=True, drop_last=True):
        self.dataset = self.dataset_class(split)
        collate_fn = lambda *params: fn_tokenizer(*params, tokenizer=tokenizer, num_bin_classes=1, special_tokens_ignore_label=VUAMetaphorDataset.IGNORE_LABEL)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=drop_last,
                                     shuffle=shuffle)
        return self.dataloader

    def describe(self):
        print('VUA Metaphor Classification Task')

    def _calculate_pos_weights(self):
        df = VUAMetaphorDataset('train').df
        num_tokens = df.sentence_txt.str.split().str.len().sum() - df.ignore_index.str.len().sum()
        num_positive = df.sentence_txt.str.count('M_').sum()
        num_negatives = num_tokens - num_positive
        pos_weights = [num_negatives / num_positive]
        return torch.tensor(pos_weights)


class VUAMetaphorAllPoSSharedTask(VUAMetaphorTask):
    NAME = 'VUAMetaphorAllPoS'
    DATASET_CLASS = VUAMetaphorSharedTaskAllPOSDataset


class VUAMetaphorVerbSharedTask(VUAMetaphorTask):
    NAME = 'VUAMetaphorVerbs'
    DATASET_CLASS = VUAMetaphorSharedTaskVerbDataset


class TaskFactory(object):
    task_map = {
        'Metaphor': VUAMetaphorTask,
        'MetaphorSTAll': VUAMetaphorAllPoSSharedTask,
        'MetaphorSTVerbs': VUAMetaphorVerbSharedTask,
        'Propaganda': PropagandaFLCTask,
        'Propaganda_LoadedLanguage': PTCLoadedLanguageTask,
        'Propaganda_NameCalling': PTCNameCallingLabelingTask,
        'News': PropagandaFLCTask,
        'News_LoadedLanguage': PTCLoadedLanguageTask,
        'News_NameCalling': PTCNameCallingLabelingTask,
        'Memes': MemesTask,
        'Memes_LoadedLanguage': MemesLoadedLanguageTask,
        'Memes_NameCalling': MemesNameCallingLabelingTask
    }

    @staticmethod
    def get_training_tasks(args):
        dims = 1024 if 'large' in args.encoder else 768
        tasks = []
        for task_name in args.training_tasks:
            task_class = TaskFactory.task_map.get(task_name)
            assert task_class
            tasks.append(task_class(cls_dim=dims, pos_weight=args.pos_weight))
        return tasks

    @staticmethod
    def get_validation_task(args):
        assert args.validation_task in args.training_tasks
        dims = 1024 if 'large' in args.encoder else 768
        task_class = TaskFactory.task_map.get(args.validation_task)
        return task_class(cls_dim=dims, pos_weight=args.pos_weight)

    @staticmethod
    def get_evaluation_task(args):
        dims = 1024 if 'large' in args.encoder else 768
        task_class = TaskFactory.task_map.get(args.evaluation_task)
        return task_class(cls_dim=dims, pos_weight=args.pos_weight)

    @staticmethod
    def get_training_and_validation_tasks(args):
        task_names = list()
        task_names += args.training_tasks
        if args.validation_task not in task_names:
            task_names.apend(args.validation_task)
        dims = 1024 if 'large' in args.encoder else 768
        return [(TaskFactory.task_map.get(task_name))(cls_dim=dims, pos_weight=args.pos_weight) for task_name in task_names]
