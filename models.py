import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
from copy import deepcopy


class Encoder(nn.Module):
    # TODO: convert n_outputs to an array to handle MTL
    def __init__(self, config):
        """
        Composed by the BERT base-uncased model to which we attach a set of
        fully-connected layers.
        """
        super(Encoder, self).__init__()
        # BERT
        self.unfreeze_num = config.unfreeze_num
        # TODO: consider add_pooling_layer=False
        if config.encoder.startswith('bert'):
            self.base_model = BertModel.from_pretrained(config.encoder)
        else:
            self.base_model = RobertaModel.from_pretrained(config.encoder)
        self.base_model.requires_grad_(False)
        for block in self.base_model.encoder.layer[-config.unfreeze_num:]:
            for params in block.parameters():
                params.requires_grad = True

    def forward(self, inputs, attention_mask=None, task_pos=0):
        outputs = self.base_model(inputs, attention_mask=attention_mask)[0]
        return outputs

    def get_trainable_params(self):
        # Copy instance of model to avoid mutation while training
        bert_model_copy = deepcopy(self.base_model)

        # Delete frozen layers from model_copy instance, save state_dicts
        state_dicts = {'unfreeze_num': self.unfreeze_num}
        for i in range(1, self.unfreeze_num+1):
            state_dicts['bert_l_-{}'.format(i)] = bert_model_copy.encoder.layer[-i].state_dict()
        return state_dicts

    def load_trainable_params(self, state_dicts):
        unfreeze_num = state_dicts['unfreeze_num']
        # Overwrite last n BERT blocks, overwrite MLP params
        for i in range(1, unfreeze_num + 1):
            self.base_model.encoder.layer[-i].load_state_dict(state_dicts['bert_l_-{}'.format(i)])

    def save_model(self, snapshot_path):
        state_dicts = self.get_trainable_params()
        torch.save(state_dicts, snapshot_path)

    def load_model(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.load_trainable_params(checkpoint)


class SLClassifier(nn.Module):
    """
    Class for Single-Layer Classifier
    """
    def __init__(self, input_dim=768, target_dim=2):
        super(SLClassifier, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(input_dim, target_dim)
            )

    def forward(self, input):
        return self.network(input)


class MultiTaskLearner(nn.Module):
    def __init__(self, config):
        super(MultiTaskLearner, self).__init__()
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, inputs, task_name=None, attention_mask=None):
        task_module_name = 'task_{}'.format(task_name)
        assert task_module_name in self._modules

        encoded = self.encoder(inputs, attention_mask=attention_mask)
        encoded = self.dropout(encoded)
        classifier = self._modules[task_module_name]
        return classifier(encoded)

    def add_task_classifier(self, task_name, classifier):
        assert issubclass(type(classifier), nn.Module)
        self.add_module('task_{}'.format(task_name), classifier)

    def save_model(self, snapshot_path):
        state_dicts = self.encoder.get_trainable_params()
        for module in self._modules:
            if 'task' in module:
                state_dicts[module+'_state_dict'] = self._modules[module].state_dict()
        torch.save(state_dicts, snapshot_path)

    def load_model(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.encoder.load_trainable_params(checkpoint)
        for module in self._modules:
            if 'task' in module:
                if module+'_state_dict' in checkpoint:
                    self._modules[module].load_state_dict(checkpoint[module+'_state_dict'])
                else:
                    print(f'Warning: No parameters for classifier "{module}" on loaded model; ' +
                          'classifier weights will be initialized randomly.')
