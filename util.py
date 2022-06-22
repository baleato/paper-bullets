from argparse import ArgumentParser
import random

from datetime import datetime
import numpy as np
import truecase
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast
import configargparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)

    # When running on the CuDNN backend two further options must be set for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


TASK_NAMES = [
    'Metaphor', 'MetaphorSTAll', 'MetaphorSTVerbs',
    'News', 'News_LoadedLanguage', 'News_NameCalling',
    'Memes', 'Memes_LoadedLanguage', 'Memes_NameCalling'
]


def get_args(args=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pos_weight', action='store_true')
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--kfold_seed', type=int, default=0)
    parser.add_argument('--encoder', type=str, default='roberta-base', choices=[
        'bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased',
        'roberta-base', 'roberta-large'
    ])
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--csv_results_file', type=str, default='results.csv')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--unfreeze_num', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--bert_lr', type=float, default=None)
    parser.add_argument('--head_lr', type=float, default=None)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--training_tasks', nargs='*', choices=TASK_NAMES,
                        default=['News'])
    parser.add_argument('--validation_task', type=str, default='News', choices=TASK_NAMES)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, choices=['constant', 'linear', 'cosine'], default='cosine')
    parser.add_argument('--weight_decay', type=float, default=0)
    # Multi-task specific
    parser.add_argument('--task_sampling_ratio', nargs='*', type=float)
    parser.add_argument('--epoch_factor', nargs='*', type=float)
    parser.add_argument('--alignment_strategy', type=str, default='min1', choices=['min1', 'max1', 'prop'],
                        help='''Used to calculate how many batches fit into the current epoch.
        - "prop" (proportional): the size of the epoch is calculated as the dot product between the task sampling
          probability and the number of batches for that task.
        - "min1": limits the total number of batches per task in the epoch to the number of batches of the smallest
          task. Sampling ratios are not taking into consideration.
        - "max1": limits the total number of batches per task in the epoch to the number of batches of the smallest
          task. Sampling ratios are not taking into consideration.''')
    parser.add_argument('--loss_task_factor', nargs='*', type=float)
    parser.add_argument('--train_ignored_labels', action='store_true', default=True,
                        help='Whether to train the tokens the task flagged to ignore '
                             '(e.g. non content words for Metaphor ALL-POS shared task). '
                             'Enabled by default, use --no-train_ignored_labels to '
                             'turn off this setting.'
                             )
    parser.add_argument('--no-train_ignored_labels', action='store_true')
    args = parser.parse_args(args)
    if args.no_train_ignored_labels:
        args.train_ignored_labels = False
    return args


def get_eval_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load_model', type=str, default='')
    # TODO: remove dropout, unfreeze_num and encoder
    #   Now required due args being passed to the model
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--unfreeze_num', type=int, default=12)
    parser.add_argument('--encoder', type=str, default='bert-base-cased', choices=[
        'bert-base-cased', 'bert-base-uncased', 'bert-large-cased', 'bert-large-uncased',
        'roberta-base', 'roberta-large'
    ])
    parser.add_argument('--evaluation_task', type=str, default='News', choices=TASK_NAMES)
    parser.add_argument('--pos_weight', action='store_true')
    args = parser.parse_args(args)
    return args


def get_pytorch_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    return device


def get_run_identifier(args, timestamp=datetime.now()):
    number_to_str = lambda n: str(n).replace('0.', '.') if int(n) != n else str(int(n))
    training_tasks = []
    task_sampling_ratio = args.task_sampling_ratio if args.task_sampling_ratio else ['eq'] * len(args.training_tasks)
    for task_index, task_name in enumerate(args.training_tasks):
        sampling_ratio = number_to_str(args.task_sampling_ratio[task_index]) if args.task_sampling_ratio else 'e'
        epoch_factor = number_to_str(args.epoch_factor[task_index]) if args.epoch_factor else '-'
        loss_factor = number_to_str(args.loss_task_factor[task_index]) if args.loss_task_factor else '-'
        is_val_task = args.validation_task == task_name
        training_tasks.append(
            f'SR{sampling_ratio}_LF{loss_factor}_EF{epoch_factor}_{task_name if is_val_task else task_name.lower()}'
        )
    return 'mtl{pretrained}_{training_tasks}_{time}_{prefix}_b{batch_size}_u{unfreeze_num}_' \
           'd{dropout}_blr{bert_lr}_hlr{head_lr}_p{patience}_w{warmup}_lrS{lr_scheduler}_pw{pos_weight}_wd{weight_decay}_{encoder}_s{seed}'.format(
        training_tasks=f'AS{args.alignment_strategy}_' + '+'.join(training_tasks),
        pretrained='PT' if args.load_model else '',
        time=timestamp.strftime('%y%m%d-%H%M%S'),
        prefix=args.prefix,
        batch_size=args.batch_size,
        unfreeze_num=args.unfreeze_num,
        dropout=number_to_str(args.dropout),
        head_lr=args.head_lr if args.head_lr else args.lr,
        bert_lr=args.bert_lr if args.bert_lr else args.lr,
        encoder=args.encoder,
        patience=args.patience,
        warmup=number_to_str(args.warmup),
        lr_scheduler=args.lr_scheduler[:3],
        pos_weight='Y' if args.pos_weight else 'N',
        weight_decay=number_to_str(args.weight_decay),
        seed=args.seed
    )


def get_tokenizer(encoder):
    if encoder.startswith('bert'):
        tokenizer = BertTokenizerFast.from_pretrained(encoder)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(encoder)
    return tokenizer


def get_true_case(s_in):
    # The true-case library alters whitespaces in the output (e.g. '\n\n' is converted to a single whitespace)'.
    # This could cause the start/end of target spans not to match the original text. We
    s_true_case = truecase.get_true_case(s_in)
    pos_in, pos_out = 0, 0
    s_out = ''
    while pos_in < len(s_in):
        if pos_out < len(s_true_case) and s_in[pos_in].lower() == s_true_case[pos_out].lower():
            s_out += s_true_case[pos_out]
            pos_out += 1
            pos_in += 1
        else:
            if pos_out < len(s_true_case) and s_true_case[pos_out] == ' ':
                pos_out += 1
            else:
                s_out += s_in[pos_in]
                pos_in += 1
    assert s_in.lower() == s_out.lower(), 'Original and true cased versions should be equal when lowered:' \
                                          f'\n=>{repr(s_in)}\n=>{repr(s_out)}'
    return s_out


if __name__ == '__main__':
    print(get_args())
