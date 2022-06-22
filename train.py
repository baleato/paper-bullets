import json
import os
import time
import glob
from datetime import datetime, timedelta
from itertools import chain
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,\
    get_constant_schedule_with_warmup

from eval import evaluate
from generate_predictions import write_predictions
from util import get_args, get_pytorch_device, get_tokenizer, set_seed, get_run_identifier
from tasks import *
from torch.utils.tensorboard import SummaryWriter
from models import MultiTaskLearner


def train(tasks, model, tokenizer, args, device):
    run_dsc = get_run_identifier(args)
    print('run_dsc: ' + run_dsc)
    # Define logging
    os.makedirs(args.save_path.replace('~', os.environ['HOME']), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_path, 'runs', run_dsc))

    header = '      Time                     Task  Iteration   Progress   Epoch %Epoch       ' + \
             'Loss   Dev/Loss     Accuracy      Dev/Acc'
    log_template = '{:>10} {:>25} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f} {:5.0f}% ' + \
                   '{:10.6f}              {:10.6f}'
    dev_log_template = '{:>10} {:>25} {:10.0f} {:5.0f}/{:<5.0f} {:5.0f} {:5.0f}%' + \
                       '            {:10.6f}              {:12.6f}'

    print(header)
    start = time.time()

    # Define optimizers and loss function
    optimizer_bert = AdamW(params=model.encoder.base_model.parameters(),
                           lr=args.bert_lr if args.bert_lr else args.lr,
                           weight_decay=args.weight_decay)
    # TODO: don't access model internals, export function to get desired parameters
    task_classifiers_params = [model._modules[m_name].parameters() for m_name in model._modules if 'task' in m_name]
    optimizer = AdamW(params=chain(*task_classifiers_params),
                      lr=args.head_lr if args.head_lr else args.lr,
                      weight_decay=args.weight_decay)

    # initialize task sampler
    task_sampling_ratio = args.task_sampling_ratio if args.task_sampling_ratio else 'equal'
    sampler = TaskSampler(tasks, task_sampling_ratio=task_sampling_ratio, epoch_factor=args.epoch_factor,
                          alignment_strategy=args.alignment_strategy)

    # Iterate over the data
    train_iter = sampler.get_dataloader('train', tokenizer=tokenizer, batch_size=args.batch_size,
                                        shuffle=True, drop_last=True)
    train_iter_len = len(train_iter)

    assert 0 <= args.warmup < 1
    # FIXME: sampler using a proportional alignment strategy might deliver epochs of different lengths
    num_training_steps = train_iter_len * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup
    if args.lr_scheduler == 'cosine':
        lr_scheduler_fn = get_cosine_schedule_with_warmup
    elif args.lr_scheduler == 'linear':
        lr_scheduler_fn = get_linear_schedule_with_warmup
    elif args.lr_scheduler == 'constant':
        lr_scheduler_fn = get_constant_schedule_with_warmup
    else:
        assert False, f'Unknown LR scheduler type: {args.lr_scheduler}'
    scheduler_bert = lr_scheduler_fn(optimizer_bert, num_warmup_steps,
                                     num_training_steps if args.lr_scheduler != 'constant' else -1)
    scheduler = lr_scheduler_fn(optimizer, num_warmup_steps,
                                num_training_steps if args.lr_scheduler != 'constant' else -1)

    if args.loss_task_factor:
        assert len(args.loss_task_factor) == len(args.training_tasks), 'Loss factors matches the number of tasks.'

    model.train()

    # setup test model, task and episodes for evaluation
    dev_task = TaskFactory.get_validation_task(args)
    aux_tasks = [task for task in tasks if task.get_name() != dev_task.get_name()]

    best_dev_fscore, best_dev_precision, best_dev_recall, snapshot_path = -1, -1, -1, ''
    iterations, running_loss = 0, {}
    patience = args.patience
    for idx_epoch in range(args.num_epochs):
        for idx_batch, batch in enumerate(train_iter):

            # Reset .grad attributes for weights
            optimizer_bert.zero_grad()
            optimizer.zero_grad()

            # Extract the sentence_ids and target vector, send sentences to GPU
            sentences = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2]

            # Feed sentences into BERT instance, compute loss, perform backward
            # pass, update weights.
            outputs = model(sentences, sampler.get_name(), attention_mask=attention_masks)
            if args.train_ignored_labels:
                # Ignored labels have negative values. Setting them to zero they act as negative examples.
                labels[torch.where(labels < 0)] = 0
                loss = sampler.get_loss(outputs, labels.to(device, dtype=torch.float))
            else:
                # We apply a mask on the labels to discard the ones the task flagged to ignore.
                labels_mask = torch.where(labels >= 0, True, False)
                loss = sampler.get_loss(outputs[labels_mask], labels.to(device, dtype=torch.float)[labels_mask])
            if args.loss_task_factor:
                loss = loss * args.loss_task_factor[sampler.get_task_index()]
            loss.backward()
            optimizer.step()
            optimizer_bert.step()
            scheduler.step()
            scheduler_bert.step()

            current_task_name = sampler.get_name()
            running_loss.setdefault(current_task_name, [])
            running_loss.get(current_task_name).append(loss.item())
            iterations += 1
            if iterations % args.log_every == 0:
                active_loss = attention_masks == 1
                y_pred = torch.where(torch.sigmoid(outputs) > 0.5, 1, 0)
                active_y_pred = y_pred[active_loss]
                active_y_true = labels[active_loss]
                precision, recall, fscore, _ = sampler.score(y_true=active_y_true.cpu(), y_pred=active_y_pred.cpu())
                writer.add_scalar(f'train/fscore/{current_task_name}', fscore, iterations)
                writer.add_scalar(f'train/precision/{current_task_name}', precision, iterations)
                writer.add_scalar(f'train/recall/{current_task_name}', recall, iterations)
                # Send to tensor board the loss for every training task
                for task_name in running_loss.keys():
                    writer.add_scalar(f'train/Loss/{task_name}',
                                      sum(running_loss.get(task_name)) / len(running_loss.get(task_name)),
                                      iterations)
                writer.flush()
                print(log_template.format(
                    str(timedelta(seconds=int(time.time() - start))),
                    sampler.get_name(),
                    iterations,
                    idx_batch + 1, train_iter_len,
                    idx_epoch + 1,
                    (idx_batch + 1) / train_iter_len * 100,
                    sum(running_loss.get(current_task_name)) / len(running_loss.get(current_task_name)), fscore))
                running_loss = {}

        # ============================ EVALUATION ============================
        model.eval()
        dev_precision, dev_recall, dev_fscore, dev_loss, _, scores_per_class = \
            evaluate(model, dev_task, 'dev', tokenizer, device)
        print(dev_log_template.format(
            str(timedelta(seconds=int(time.time() - start))),
            dev_task.get_name(),
            iterations,
            idx_batch + 1, train_iter_len,
            idx_epoch + 1,
            (idx_batch + 1) / train_iter_len * 100,
            dev_loss, dev_fscore))

        writer.add_scalar(f'dev/fscore/{dev_task.get_name()}', dev_fscore, iterations)
        writer.add_scalar(f'dev/precision/{dev_task.get_name()}', dev_precision, iterations)
        writer.add_scalar(f'dev/recall/{dev_task.get_name()}', dev_recall, iterations)
        writer.add_scalar(f'dev/Loss/{dev_task.get_name()}', dev_loss, iterations)
        if scores_per_class:
            for class_name in scores_per_class.keys():
                _precision = scores_per_class.get(class_name).get('precision')
                _recall = scores_per_class.get(class_name).get('recall')
                _f1 = scores_per_class.get(class_name).get('f1')
                writer.add_scalar(f'dev/precision/{class_name}', _precision, iterations)
                writer.add_scalar(f'dev/recall/{class_name}', _recall, iterations)
                writer.add_scalar(f'dev/fscore/{class_name}', _f1, iterations)

        for train_task, prob in zip(tasks, train_iter.task_probs):
            writer.add_scalar(f'train/SamplingProb/{train_task.get_name()}', prob, iterations)

        for aux_task in aux_tasks:
            aux_precision, aux_recall, aux_fscore, aux_loss, _, _ = \
                evaluate(model, aux_task, 'dev', tokenizer, device)
            writer.add_scalar(f'aux/{aux_task.get_name()}/fscore', aux_fscore, iterations)
            writer.add_scalar(f'aux/{aux_task.get_name()}/precision', aux_precision, iterations)
            writer.add_scalar(f'aux/{aux_task.get_name()}/recall', aux_recall, iterations)
            writer.add_scalar(f'aux/{aux_task.get_name()}/Loss', aux_loss, iterations)
            print(f'AUX {aux_task.get_name()}: P{aux_precision:.6f} R{aux_recall:.6f} F1{aux_fscore:.6f}')

        if best_dev_fscore < dev_fscore:
            patience = args.patience
            best_dev_fscore, best_dev_precision, best_dev_recall = dev_fscore, dev_precision, dev_recall
            snapshot_prefix = os.path.join(args.save_path, 'best_' + run_dsc)
            snapshot_path = (
                    snapshot_prefix +
                    '_it{iterations}_ep{epoch}_P{precision:.4f}_R{recall:.4f}_F{fscore:.4f}.pt'
            ).format(
                iterations=iterations,
                epoch=idx_epoch + 1,
                precision=best_dev_precision,
                recall=best_dev_recall,
                fscore=best_dev_fscore
            )
            model.save_model(snapshot_path)
            # Keep only the best snapshot
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)
        else:
            patience -= 1
            if not patience:
                print('Stop after running out of patience')
                break;

    writer.close()

    # Rename tensorboard log file to store prediction, recall and f-score of the best model
    old_filename = os.path.join(args.save_path, 'runs', run_dsc)
    new_filename = old_filename + '_P{precision:.4f}_R{recall:.4f}_F{fscore:.4f}'.format(
        precision=best_dev_precision,
        recall=best_dev_recall,
        fscore=best_dev_fscore
    )
    print('Metrics file: ' + new_filename)
    os.rename(old_filename, new_filename)
    return best_dev_precision, best_dev_recall, best_dev_fscore, snapshot_path


if __name__ == '__main__':
    args = get_args()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)
    set_seed(args.seed)

    print('Loading Tokenizer..')
    tokenizer = get_tokenizer(args.encoder)

    tasks = TaskFactory.get_training_and_validation_tasks(args)
    model = MultiTaskLearner(args)
    for task in tasks:
        print(f'Adding classifier for task: {task.get_name()}')
        model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
    if args.load_model:
        print("Loading models from snapshot")
        model.load_model(args.load_model, device)
    model.to(device)
    results = train(TaskFactory.get_training_tasks(args), model, tokenizer, args, device)
    best_dev_precision, best_dev_recall, best_dev_fscore, snapshot_path = results
    print('=' * 10 + ' DEV SET ' + '=' * 10)
    print(f'Precision: {best_dev_precision:.6f}')
    print(f'Recall: {best_dev_recall:.6f}')
    print(f'F1: {best_dev_fscore:.6f}')

    if snapshot_path:
        model.load_model(snapshot_path, device)
    validation_task = TaskFactory.get_validation_task(args)
    precision, recall, fscore, loss, conf_matrix, dev_scores_per_class = evaluate(model, validation_task, 'dev', tokenizer, device)
    print('=' * 10 + ' DEV SET (AGAIN) ' + '=' * 10)
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'F1: {fscore:.6f}')

    if snapshot_path:
        model.load_model(snapshot_path, device)
    precision, recall, fscore, loss, conf_matrix, test_scores_per_class = evaluate(model, validation_task, 'test', tokenizer, device)
    print('=' * 10 + ' TEST SET ' + '=' * 10)
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'F1: {fscore:.6f}')
    print(f'Loss: {loss}')
    print(f'Confusion matrix:')
    print(conf_matrix)

    auxiliary_tasks_metrics = {}
    for aux_task in [task for task in tasks if task.NAME != validation_task.NAME]:
        aux_precision, aux_recall, aux_fscore, aux_loss, _, _ = \
            evaluate(model, aux_task, 'test', tokenizer, device)
        print('=' * 10 + f' AUX TASK {aux_task.NAME} ' + '=' * 10)
        print(f'Precision: {aux_precision:.6f}')
        print(f'Recall: {aux_recall:.6f}')
        print(f'F1: {aux_fscore:.6f}')
        auxiliary_tasks_metrics.setdefault(aux_task.get_name(), {
            'precision': aux_precision,
            'recall': aux_recall,
            'f1': aux_fscore
        })

    csv_file = os.path.join(args.save_path, args.csv_results_file)
    with open(csv_file, 'a') as csv_out:
        csv_out.write(
            # index, precision, recall, fscore
            f'{str(datetime.now())},{precision},{recall},{fscore},{best_dev_precision},{best_dev_recall},{best_dev_fscore},{snapshot_path},"' +
            json.dumps(vars(args)).replace('"', '""') + '","' +
            json.dumps(test_scores_per_class).replace('"', '""') + '","' +
            json.dumps(dev_scores_per_class).replace('"', '""') + '","' +
            json.dumps(auxiliary_tasks_metrics).replace('"', '""') +
            '"\n'
        )

    write_predictions(model, validation_task, tokenizer, device, snapshot_path)
