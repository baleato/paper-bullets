from datetime import datetime
import time
import json
import torch
from torch import Tensor
from transformers import BatchEncoding

from data import PTC, MPTC
from eval import _combine_token_to_words
from models import MultiTaskLearner
from tasks import MemesTask, TaskFactory
from util import get_pytorch_device, get_eval_args, get_tokenizer


def expand_tags_to_whole_words(inputs: BatchEncoding, tags: Tensor):
    tags_out = torch.zeros(tags.shape, dtype=int)
    num_classes = tags.shape[1]
    for technique_idx in range(0, num_classes):
        for token_idx in range(0, len(tags)):
            word_id = inputs.token_to_word(token_idx)
            if word_id is not None and tags[token_idx][technique_idx]:
                for index, _word_id in enumerate(inputs.word_ids()):
                    if word_id == _word_id:
                        tags_out[index][technique_idx] = 1
    return tags_out


def get_technique_spans(inputs: BatchEncoding, tags: Tensor):
    tags = expand_tags_to_whole_words(inputs, tags)
    num_classes = tags.shape[1]
    start_tag = 0
    end_tag = 0
    techniques = []
    for technique_idx in range(0, num_classes):
        curr_technique_active, prev_technique_active = False, None
        for token_idx in range(0, len(tags)):
            if inputs.token_to_word(token_idx) == None:
                continue
            start, end = inputs.token_to_chars(token_idx)
            curr_technique_active = int(tags[token_idx][technique_idx])
            if curr_technique_active == prev_technique_active:
                if curr_technique_active:
                    end_tag = end
            else:
                if prev_technique_active:
                    if end_tag > start_tag:
                        # RoBERTa classify empty tokens, we discard those as the
                        # scorer script fails when the start and end position
                        # are the same
                        techniques.append((technique_idx, start_tag, end_tag))
                else:
                    start_tag = start
                    end_tag = end
            prev_technique_active = curr_technique_active
        if curr_technique_active:
            techniques.append((technique_idx, start_tag, end))
    return techniques


def get_score(labels_a, labels_b):
    if not labels_a:
        return 0
    _sum = 0
    for technique_a, start_a, end_a in labels_a:
        for technique_b, start_b, end_b in labels_b:
            if technique_b == technique_a and end_b > start_b:
                s = set(range(start_a, end_a))
                t = set(range(start_b, end_b))
                h = end_b - start_b
                c = len(s.intersection(t)) / h
                _sum += c
    return _sum / len(labels_a)


def get_precision(gold, pred):
    return get_score(gold, pred)


def get_recall(gold, pred):
    return get_score(pred, gold)


def get_precision_recall_fscore(gold, pred):
    precision = get_precision(gold, pred)
    recall = get_recall(gold, pred)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * (precision * recall) / (precision + recall)
    return precision, recall, fscore


def write_predictions(model, task, tokenizer, device, filename=''):
    df = task.dataset_class('test').df
    i, total = 0, len(df)
    output = []
    t_start = datetime.now().timestamp()
    f_json = open(filename + '.json', 'w')
    f_txt = open(filename + '.txt', 'w')
    model.eval()
    if isinstance(task, MemesTask):
        transform = lambda e: (e.get('technique'), e.get('start'), e.get('end'))
        convert_label = lambda technique_idx: MPTC.LABELS[technique_idx]
    else:
        transform = lambda e: (PTC.LABELS[e[0]], e[1][0], e[1][1])
        convert_label = lambda technique_idx: PTC.LABELS[technique_idx]
    precisions, recalls, len_golds, len_preds = [], [], [], []
    for example in df.itertuples():
        if task.get_name().startswith('Propaganda'):
            _, id, text, labels, offset = list(example)
        else:
            _, id, text, labels = list(example)
            offset = 0
        encoded = tokenizer(text, return_tensors='pt')
        outputs = model(encoded.input_ids.to(device), task.get_name())
        logits = torch.sigmoid(outputs)
        y_pred = torch.where(logits > 0.5, 1, 0)
        techniques = get_technique_spans(encoded, y_pred[0])

        golds = [transform(label) for label in labels]
        preds = [(convert_label(technique_idx), start, end) for technique_idx, start, end in techniques]
        precisions.append(get_precision(golds, preds))
        len_golds.append(len(golds))
        recalls.append(get_recall(golds, preds))
        len_preds.append(len(preds))
        example_predictions = {
            "id": id,
            "sentence": text,
            "offset": offset,
            "labels": []
        }
        # TODO: integrate
        if 'task_VUAMetaphorAllPoS' in model._modules:
            outputs_metaphor = model(encoded.input_ids.to(device), 'VUAMetaphorAllPoS')
            logits_metaphor = torch.sigmoid(outputs_metaphor)
            y_pred_metaphor = torch.where(logits_metaphor > 0.5, 1, 0)
            y_pred_metaphor[0] = expand_tags_to_whole_words(encoded, y_pred_metaphor[0])
            y_pred_metaphor_words = _combine_token_to_words(encoded, y_pred_metaphor, tokenizer.pad_token_id)
            metaphor_text = text
            for word_index, word_label in reversed(list(enumerate(y_pred_metaphor_words[0]))):
                if word_label:
                    # Metaphor
                    start_word, _end_word = encoded.word_to_chars(word_index)
                    metaphor_text = metaphor_text[:start_word] + 'M_' + metaphor_text[start_word:]
            example_predictions['sentence_metaphor'] = metaphor_text
        for technique_idx, start, end in techniques:
            technique = task.dataset_class.LABELS[technique_idx]
            example_predictions["labels"].append({
                "start": offset + start,
                "end": offset + end,
                "technique": technique,
                "text_fragment": text[start:end]
            })
            line = f'{id}\t{technique}\t{offset + start}\t{offset + end}\n'
            f_txt.write(line)
        output.append(example_predictions)

        i += 1
        if i % 20 == 0:
            duration = datetime.now().timestamp() - t_start
            perc_passed = i / total
            expected_end = duration / perc_passed * (1 - perc_passed)
            ty_res = time.gmtime(expected_end)
            res = time.strftime("%H:%M:%S", ty_res)
            print(f'{i / total * 100:.2f}% ETA: {res}', end='\r')
    precisions = torch.tensor(precisions)
    recalls = torch.tensor(recalls)
    len_golds = torch.tensor(len_golds)
    len_preds = torch.tensor(len_preds)
    precision = sum(precisions * len_golds) / sum(len_preds)
    recall = sum(recalls * len_preds) / sum(len_golds)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    print(
        f"INFO: Precision={sum(precisions * len_golds):.6f}/{sum(len_preds)}\tRecall={sum(recalls * len_preds):.6f}/{sum(len_golds)}")
    print(f"\nF1={fscore:.6f}")
    print(f"Precision={precision:.6f}")
    print(f"Recall={recall:.6f}")
    # print(json.dumps(output, indent=True))
    f_json.write(json.dumps(output, indent=True))
    f_json.close()
    f_txt.close()


if __name__ == '__main__':
    args = get_eval_args()
    device = get_pytorch_device(args)
    task = TaskFactory.get_evaluation_task(args)
    model = MultiTaskLearner(args)
    model.to(device)
    model.add_task_classifier(task.get_name(), task.get_classifier().to(device))
    if args.load_model:
        print(f"Loading models from: {args.load_model}")
        model.load_model(args.load_model, device)
        filename = args.load_model + '_predictions'
    else:
        print('Warning! No pre-train model given. Random initialization')
        filename = 'random_initialized_model_predictions'

    tokenizer = get_tokenizer(args.encoder)

    write_predictions(model, task, tokenizer, device, filename)
