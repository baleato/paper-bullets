from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from torch import Tensor
from transformers import BatchEncoding

from models import MultiTaskLearner
from tasks import *
from util import get_pytorch_device, get_eval_args, get_tokenizer


def _convert_to_start_end(binary_tensor):
    """Auxiliary function to convert a binary list in positive spans.
    Example: f([0,0,1,1,1,0,0,1,0]) = [(2,5), (7,8)]
    """
    acc = []
    prev_val = 0
    track_index = -1
    for index, value in enumerate(binary_tensor):
        if value:
            if not prev_val:
                track_index = index
        else:
            if prev_val:
                acc.append((track_index, index))
        prev_val = value
    if prev_val:
        acc.append((track_index, len(binary_tensor)))
    return acc


def _expand_tags_to_whole_words(inputs: BatchEncoding, tags: Tensor):
    """Post-processing step were we extend the predicted propaganda tokens to encompass the whole word."""
    tags_out = torch.zeros(tags.shape, dtype=int)
    batch_size = tags.shape[0]
    num_classes = tags.shape[2]
    for batch_index in range(0, batch_size):
        for technique_idx in range(0, num_classes):
            for token_idx in range(0, len(tags[batch_index])):
                word_id = inputs.token_to_word(batch_index, token_idx)
                if word_id is not None and tags[batch_index][token_idx][technique_idx]:
                    for index, _word_id in enumerate(inputs.word_ids(batch_index)):
                        if word_id == _word_id:
                            tags_out[batch_index][index][technique_idx] = 1
    return tags_out


def get_cumulative_score(labels_a, labels_b, num_classes):
    """Returns the cumulative partial overlap of fragments used to calculate precision and recall (SemEval2021 Task6):
        - C(s,t,h)=\frac{|(s\bigcap{t})|}{h}\delta(l(s), l(t))
    """
    _sum = np.zeros(num_classes)
    _len = np.zeros(num_classes)
    for technique_a, (start_a, end_a) in labels_a:
        _len[technique_a] += 1
        for technique_b, (start_b, end_b) in labels_b:
            if technique_b == technique_a:
                s = set(range(start_a, end_a))
                t = set(range(start_b, end_b))
                h = end_a - start_a
                intersection = len(s.intersection(t))
                c = intersection / h
                _sum[technique_a] += c
    return _sum, _len


def _transform_as_labels(tags: Tensor, inputs: BatchEncoding):
    """Converts the labeled tokens to a list with elements in the following format: (technique_id, (start, end))"""
    batch_size, _, num_classes = tags.shape
    labels_batch = []
    for batch_index in range(0, batch_size):
        labels_example = []
        for technique_idx in range(0, num_classes):
            spans_in_tokens = _convert_to_start_end(tags[batch_index, :, technique_idx])
            spans_in_chars = [(inputs.token_to_chars(batch_index, start)[0], inputs.token_to_chars(batch_index, end - 1)[1]) for start, end in spans_in_tokens]
            label_spans = [(technique_idx, (start, end)) for start, end in spans_in_chars if end > start]
            labels_example += label_spans
        labels_batch.append(labels_example)
    return labels_batch


def evaluate(model, task, split, tokenizer, device, batch_size=16):
    # FIXME: delegate evaluation to the task
    if isinstance(task, VUAMetaphorTask):
        return evaluate_metaphor(model, task, split, tokenizer, device, batch_size=batch_size)
    dev_iter = task.get_dataloader(split, tokenizer=tokenizer, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    sum_loss = 0
    y_all_gold, y_all_preds = torch.empty(0, dtype=int), torch.empty(0, dtype=int)
    iter_len = len(dev_iter)
    num_classes = len(task.dataset.LABELS)
    cumulative_s_prec, cumulative_s_rec = torch.zeros(num_classes), torch.zeros(num_classes)
    list_len_gold, list_len_preds = torch.zeros(num_classes), torch.zeros(num_classes)
    with torch.no_grad():
        for batch_idx, dev_batch in enumerate(dev_iter):
            sentences = dev_batch[0].to(device)
            attention_masks = dev_batch[1].to(device)
            labels = dev_batch[2]
            tokenizer_results = dev_batch[3]

            outputs = model(sentences, task.get_name(), attention_mask=attention_masks)

            # Loss
            batch_dev_loss = task.get_loss(outputs, labels.to(device, dtype=torch.float))
            sum_loss += batch_dev_loss.item()

            y_pred = torch.where(torch.sigmoid(outputs) > 0.5, 1, 0)
            y_pred = _expand_tags_to_whole_words(tokenizer_results, y_pred) # Post-processing
            active_loss = attention_masks == 1

            pred_labels = _transform_as_labels(y_pred, tokenizer_results)
            for i in range(0, y_pred.shape[0]):
                _, gold_labels = task.dataset.__getitem__(batch_idx * batch_size + i)
                _sum, _len = get_cumulative_score(pred_labels[i], gold_labels, num_classes)
                cumulative_s_prec += _sum
                list_len_preds += _len
                _sum, _len = get_cumulative_score(gold_labels, pred_labels[i], num_classes)
                cumulative_s_rec += _sum
                list_len_gold += _len

            active_y_pred = y_pred[active_loss]
            active_y_true = labels[active_loss]

            y_all_gold = torch.cat((y_all_gold, active_y_true.cpu()))
            y_all_preds = torch.cat((y_all_preds, active_y_pred.cpu()))

            print(f'\r{batch_idx / iter_len * 100:.2f}%', end='')

        print('\r', end='')

    precision_per_class = torch.nan_to_num(cumulative_s_prec / list_len_preds)
    recall_per_class = torch.nan_to_num(cumulative_s_rec / list_len_gold)
    fscore_per_class = torch.nan_to_num(2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class))
    precision = torch.sum(cumulative_s_prec) / torch.sum(list_len_preds) if torch.sum(list_len_preds) > 0 else 0
    recall = torch.sum(cumulative_s_rec) / torch.sum(list_len_gold)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    conf_matrix = multilabel_confusion_matrix(y_true=y_all_gold, y_pred=y_all_preds, labels=range(len(task.dataset.LABELS)))
    loss = sum_loss / iter_len
    # for i, label_name in enumerate(task.dataset.LABELS):
    #     print(f'F1_{label_name}: P={precision_per_class[i]:.6f} R={recall_per_class[i]:.6f} F1={fscore_per_class[i]:.6f}')
    scores_per_class = {}
    for i, label_name in enumerate(task.dataset.LABELS):
        scores_per_class[label_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(fscore_per_class[i])
        }
    return float(precision), float(recall), float(fscore), loss, conf_matrix, scores_per_class


def _combine_token_to_words(tokenizer_results: BatchEncoding, tokens: Tensor, pad_id):
    """"
    Uses the tokenizer's output to combine all the tokens that belong to the same word into one element. To keep the
    same length for all sequences the pad_id given is used.
    Useful to evaluation metrics that are word-based instead of token-based.
    """
    results = []
    max_length = 0
    for seq_id in range(len(tokenizer_results.input_ids)):
        tmp = []
        last_word_index = -1
        for token_index, word_index in enumerate(tokenizer_results.word_ids(seq_id)):
            if word_index is None:
                # Discard special token positions
                pass
            elif last_word_index < word_index:
                tmp.append(tokens[seq_id, token_index])
                last_word_index = word_index
            else:
                # Discard subsequent inputs for the same token
                pass
                assert tokens[seq_id, token_index] == tmp[-1],\
                    'All tokens for the same word have the same representation'
        if max_length < len(tmp):
            max_length = len(tmp)
        results.append(tmp)
    # Pad sequences
    for i in range(len(results)):
        while len(results[i]) < max_length:
            results[i].append(pad_id)
    return torch.tensor(results)


# FIXME: integrate in one evaluation function
def evaluate_metaphor(model, task, split, tokenizer, device, batch_size=16):
    dev_iter = task.get_dataloader(split, tokenizer=tokenizer, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    sum_loss = 0
    iter_len = len(dev_iter)
    y_all_gold, y_all_preds = torch.empty(0, dtype=int), torch.empty(0, dtype=int)
    with torch.no_grad():
        for batch_idx, dev_batch in enumerate(dev_iter):
            sentences = dev_batch[0].to(device)
            attention_masks = dev_batch[1].to(device)
            labels = dev_batch[2]
            tokenizer_results = dev_batch[3]

            # TODO: allow model to return both token and word based results
            outputs = model(sentences, task.get_name(), attention_mask=attention_masks)

            # Loss
            batch_dev_loss = task.get_loss(outputs, labels.to(device, dtype=torch.float))
            sum_loss += batch_dev_loss.item()

            y_pred = torch.where(torch.sigmoid(outputs) > 0.5, 1, 0)
            y_pred = _expand_tags_to_whole_words(tokenizer_results, y_pred)  # Post-processing
            y_pred = _combine_token_to_words(tokenizer_results, y_pred, tokenizer.pad_token_id)

            # Ignored labels have negative values
            labels_mask = torch.where(labels >= 0, True, False)
            labels = _combine_token_to_words(tokenizer_results, labels, 0)
            labels_mask = _combine_token_to_words(tokenizer_results, labels_mask, False)
            assert labels_mask.shape == labels.shape == y_pred.shape
            active_y_pred = y_pred[labels_mask].view(-1)
            active_y_true = labels[labels_mask].view(-1)
            y_all_gold = torch.cat((y_all_gold, active_y_true.cpu()))
            y_all_preds = torch.cat((y_all_preds, active_y_pred.cpu()))
            print(f'\rMetaphor evaluation {batch_idx / iter_len * 100:.2f}%', end='')
        print('\r', end='')

    # F1 measure (for class metaphor)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_all_gold, y_pred=y_all_preds,
                                                                   average='binary')
    conf_matrix = multilabel_confusion_matrix(y_true=y_all_gold, y_pred=y_all_preds,
                                              labels=range(len(task.dataset.LABELS)))
    loss = sum_loss / iter_len
    scores_per_class = None # single class
    return float(precision), float(recall), float(fscore), loss, conf_matrix, scores_per_class


if __name__ == '__main__':
    args = get_eval_args()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)
    print('Loading Tokenizer..')
    tokenizer = get_tokenizer(args.encoder)

    task = TaskFactory.get_evaluation_task(args)
    model = MultiTaskLearner(args)
    print(f'Adding classifier for task: {task.get_name()}')
    model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
    if args.load_model:
        print("Loading models from snapshot")
        model.load_model(args.load_model, device)
    else:
        print('Warning! No pre-train model given. Random initialization')
    model.to(device)
    precision, recall, fscore, loss, conf_matrix = evaluate(model, task, 'test', tokenizer, device)
    print(f'Precision: {precision:.6f}')
    print(f'Recall: {recall:.6f}')
    print(f'F1: {fscore:.6f}')