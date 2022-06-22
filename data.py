from os import path

from tokenizers import Encoding
from torch.utils.data import IterableDataset, Dataset
import glob
import numpy as np
import pandas as pd
import torch

import util


def _chars_to_tokens(tokenizer_results: Encoding, word_start, word_end):
    """
    The tokenizers from HuggingFace provide a char_to_token function that returns a unique token_index (or None for
    special characters). However the tokenizer splits some characters into more than one token. E.g.:
    ```
    >> tokenizer('‘').tokens()
        ['<s>', 'âĢ', 'ĺ', '</s>']
    ```
    We'll use this auxiliary function to overcome the limitation of not having a `char_to_tokens` function in the
    tokenizer.
    """
    result = set()
    for token_id, _token in enumerate(tokenizer_results.tokens):
        token_chars = tokenizer_results.token_to_chars(token_id)
        if token_chars:
            token_start, token_end = token_chars[0], token_chars[1]
            if word_start < token_end and word_end > token_start:
                result.add(token_id)
    return list(result)


def fn_tokenizer(examples, tokenizer, num_bin_classes=1, special_tokens_ignore_label=None):
    """
    Receives examples to be tokenized.
    Params:
    - examples: array of examples. Each example is a tuple containing:
        - [0] sentence
        - [1] array containing the label and the slices (start, end) where the
              label occurs in the input sentence
    """
    sentences = [tmp[0] for tmp in examples]
    label_slices = [tmp[1] for tmp in examples]
    tokenizer_results = tokenizer(
        sentences,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt',
        max_length=512)
    token_labels = []
    for index in range(len(sentences)):
        tokenizer_result = tokenizer_results[index]
        tmp = np.zeros((len(tokenizer_result.tokens), num_bin_classes), dtype=int) # tokenizer_result.num_tokens not available in BertTokenizerFast
        if special_tokens_ignore_label is not None:
            # Set the ignore_label given to the first and last tokens
            for input_index, input_id in enumerate(tokenizer_results.input_ids[index]):
                if input_id in tokenizer.all_special_ids:
                    tmp[input_index] = special_tokens_ignore_label
        assert len(tmp) <= 512
        for label, (start, end) in label_slices[index]:
            label_token_ids = _chars_to_tokens(tokenizer_result, start, end)
            if label_token_ids:
                # The dataset contains some spans that deliver no words. F.i. the text fragment "\n" is labeled as
                # "Reductio ad hitlerum" in one instance. We discard those labels.
                token_start, token_end = min(label_token_ids), max(label_token_ids)
                if label == special_tokens_ignore_label:
                    # Ignored positions
                    tmp[token_start:token_end+1, :] = special_tokens_ignore_label
                else:
                    tmp[token_start:token_end+1, label] = 1
        token_labels.append(tmp)
    return tokenizer_results.input_ids, tokenizer_results.attention_mask, torch.tensor(
        token_labels), tokenizer_results


class VUAMetaphorDataset(Dataset):
    """
    Dataset wrapper for VUA metaphor. We use the Shared Task on Metaphor Detection
    used in NAACL 2018 and ACL 2020. This task has specific train and test splits
    but no validation split. We select 10% of the texts in the training set to
    have a stable validation set while finetuning the model.
    Each example yields:
        - Sentence
        - labels: 1 for words with metaphorical meaning; 0 otherwise
    """
    TRAINING_PARTION = [
        'a1e-fragment01', 'a1f-fragment06', 'a1f-fragment07', 'a1f-fragment08', 'a1f-fragment09',
        'a1f-fragment10', 'a1f-fragment11', 'a1f-fragment12', 'a1g-fragment26', 'a1g-fragment27',
        'a1h-fragment05', 'a1h-fragment06', 'a1j-fragment34', 'a1k-fragment02', 'a1l-fragment01',
        'a1m-fragment01', 'a1n-fragment09', 'a1n-fragment18', 'a1p-fragment01', 'a1p-fragment03',
        'a1x-fragment03', 'a1x-fragment04', 'a1x-fragment05', 'a2d-fragment05', 'a38-fragment01',
        'a39-fragment01', 'a3c-fragment05', 'a3e-fragment03', 'a3k-fragment11', 'a3p-fragment09',
        'a4d-fragment02', 'a6u-fragment02', 'a7s-fragment03', 'a7y-fragment03', 'a80-fragment15',
        'a8m-fragment02', 'a8n-fragment19', 'a8r-fragment02', 'a8u-fragment14', 'a98-fragment03',
        'a9j-fragment01', 'ab9-fragment03', 'ac2-fragment06', 'acj-fragment01', 'ahb-fragment51',
        'ahc-fragment60', 'ahf-fragment24', 'ahf-fragment63', 'ahl-fragment02', 'ajf-fragment07',
        'al0-fragment06', 'al2-fragment23', 'al5-fragment03', 'alp-fragment01', 'amm-fragment02',
        'as6-fragment01', 'as6-fragment02', 'b1g-fragment02', 'bpa-fragment14', 'c8t-fragment01',
        'cb5-fragment02', 'ccw-fragment03', 'cdb-fragment02', 'cdb-fragment04', 'clp-fragment01',
        'crs-fragment01', 'ea7-fragment03', 'ew1-fragment01', 'fef-fragment03', 'fet-fragment01',
        'fpb-fragment01', 'g0l-fragment01', 'kb7-fragment10', 'kbc-fragment13', 'kbd-fragment07',
        'kbh-fragment01', 'kbh-fragment02', 'kbh-fragment03', 'kbh-fragment09', 'kbh-fragment41',
        'kbj-fragment17', 'kbp-fragment09', 'kbw-fragment04', 'kbw-fragment11', 'kbw-fragment17',
        'kbw-fragment42', 'kcc-fragment02', 'kcf-fragment14', 'kcu-fragment02', 'kcv-fragment42'
    ]

    TESTING_PARTION = [
        'a1j-fragment33', 'a1u-fragment04', 'a31-fragment03', 'a36-fragment07', 'a3e-fragment02',
        'a3m-fragment02', 'a5e-fragment06', 'a7t-fragment01', 'a7w-fragment01', 'aa3-fragment08',
        'ahc-fragment61', 'ahd-fragment06', 'ahe-fragment03', 'al2-fragment16', 'b17-fragment02',
        'bmw-fragment09', 'ccw-fragment04', 'clw-fragment01', 'cty-fragment03', 'ecv-fragment05',
        'faj-fragment17', 'kb7-fragment31', 'kb7-fragment45', 'kb7-fragment48', 'kbd-fragment21',
        'kbh-fragment04', 'kbw-fragment09'
    ]
    LABELS = ['Metaphor']
    IGNORE_LABEL = -100
    _DEV_SPLIT_MASK = None
    def __init__(self, split='train', root='./data/VUA', filename='vuamc_corpus_all.csv'):
        super(VUAMetaphorDataset).__init__()
        self.df = pd.read_csv(root + '/' + filename)
        # Remove 13 NaN sentences
        self.df = self.df[~self.df.sentence_txt.isna()]
        if split == 'test':
            txt_ids = self.TESTING_PARTION
        else:
            txt_ids = self.TRAINING_PARTION
        condition = self.df.txt_id.isin(txt_ids)
        self.df = self.df[condition]
        if split != 'test':
            # The dataset does not provide a dev partition for validation.
            # We randomly generate from 10% of the training data. The mask is static so train and dev
            # splits are exclusive for the lifetime of the training.
            if VUAMetaphorDataset._DEV_SPLIT_MASK is None:
                VUAMetaphorDataset._DEV_SPLIT_MASK = np.random.rand(len(self.df)) < 0.1
            if split == 'dev':
                self.df = self.df[VUAMetaphorDataset._DEV_SPLIT_MASK]
            else:
                self.df = self.df[~VUAMetaphorDataset._DEV_SPLIT_MASK]
        self.df['metaphor_index'] = self.df.sentence_txt.apply(lambda s: [i for i, w in enumerate(s.split())
                                                                          if w.startswith('M_')])
        # Shared task will use this column to discard words
        self.df['ignore_index'] = self.df.sentence_txt.apply(lambda s: [])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, slice) or isinstance(idx, list) or isinstance(idx, np.ndarray):
            examples = self.df.iloc[idx].itertuples()
            return [self._decode_sentence(example) for example in examples]
        else:
            example = self.df.iloc[idx]
            cleaned_sentence, metaphor_chars = self._decode_sentence(example)
            return cleaned_sentence, metaphor_chars

    def _decode_sentence(self, example):
        """
        Returns:
        - cleaned_sentence: Sentence without 'M_' markings.
        - metaphor_chars: List with (start, end) pairs where the metaphor occurs in the cleaned_sentence
        """
        sentence = example.sentence_txt
        word_labels = [1 if i in example.metaphor_index
                       else self.IGNORE_LABEL if i in example.ignore_index
                       else 0 for i in range(len(sentence.split()))]
        # Check that all our positive words were tagged as metaphor
        assert all([word.startswith('M_') for word, label_metaphor in zip(sentence.split(), word_labels)
                    if label_metaphor == 1])
        cleaned_sentence = sentence.replace('M_', '')
        words = cleaned_sentence.split()
        cleaned_sentence = ' '.join(words) # Remove consecutive whitespaces
        position, metaphor_chars = 0, []
        for word, word_label in zip(words, word_labels):
            word_length = len(word)
            if word_label != 0:
                # There is only one label (metaphor; index 0)
                word_label_index = 0 if word_label == 1 else word_label
                metaphor_chars.append((word_label_index, (position, position + word_length)))
            position += word_length + 1
        return cleaned_sentence, metaphor_chars


class VUAMetaphorSharedTaskDataset(VUAMetaphorDataset):
    def __init__(self, split='train', root='./data/VUA', subtask='all_pos'):
        assert subtask in ['all_pos', 'verb']
        super().__init__(split, root=root)
        self._parse_and_include_shared_task_labels(split, root=root, subtask=subtask)

    def _parse_and_include_shared_task_labels(self, split, root='./data/VUA', subtask='all_pos'):
        """
            - split: train/dev/test
            - root: path to VUA data
            - subtask: all_pos or verbs
        """
        split = 'test' if split == 'test' else 'train'
        fn = f'{root}/naacl_flp_{split}_gold_labels/{subtask}_tokens.csv'
        df_labels = pd.read_csv(fn, names=['word_id', 'flag'], dtype={'flag': bool})
        df_labels['txt_id'] = df_labels.word_id.str.split('_').apply(lambda a: a[0])
        df_labels['sentence_id'] = df_labels.word_id.str.split('_').apply(lambda a: a[1])
        df_labels['word_index'] = df_labels.word_id.str.split('_').apply(lambda a: int(a[2]) - 1)
        self.df = self.df.join(df_labels.groupby(['txt_id', 'sentence_id']).agg({'word_index': list, 'flag': list}),
                               on=['txt_id', 'sentence_id'])
        # populate word_index and flag for columns without a match (NaN) with an empty array
        self.df['word_index'] = self.df.word_index.apply(lambda a: [] if type(a) == float else a)
        self.df['flag'] = self.df.flag.apply(lambda a: [] if type(a) == float else a)
        # Rename and use 0-based indexes
        self.df['metaphor_index'] = self.df.apply(
            lambda row: [index for index, flag in zip(row['word_index'], row['flag']) if flag], axis=1)
        self.df['ignore_index'] = self.df.apply(lambda row: self._get_ignore_index(row), axis=1)
        # Cleanup not targeted metaphors (remove M_ on not positive words)
        self.df['sentence_txt'] = self.df.apply(lambda row: ' '.join([
            w if i in row['metaphor_index'] else
            w.replace('M_', '') for i, w in enumerate(row['sentence_txt'].split())]), axis=1)
        assert all(self.df.sentence_txt.str.count('M_') == self.df.metaphor_index.str.len())
        # Remove columns we no longer need
        self.df.drop(columns=['word_index', 'flag'], inplace=True)


    def _get_ignore_index(self, row):
        ignore_index = []
        clean_sentence = row.sentence_txt.replace('M_', '')
        for index, _word in enumerate(row.sentence_txt.split()):
            if type(row.word_index) != list or index not in row.word_index:
                ignore_index.append(index)
        return ignore_index


class VUAMetaphorSharedTaskAllPOSDataset(VUAMetaphorSharedTaskDataset):
    def __init__(self, split='train', root='./data/VUA'):
        super().__init__(split, root=root, subtask='all_pos')


class VUAMetaphorSharedTaskVerbDataset(VUAMetaphorSharedTaskDataset):
    def __init__(self, split='train', root='./data/VUA'):
        super().__init__(split, root=root, subtask='verb')


class PTC(Dataset):
    """Propaganda Techniques Corpus"""
    LABELS = [
        'Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon', 'Black-and-White_Fallacy',
        'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
        'Name_Calling,Labeling', 'Obfuscation,Intentional_Vagueness,Confusion', 'Red_Herring',
        'Reductio_ad_hitlerum', 'Repetition', 'Slogans', 'Straw_Men', 'Thought-terminating_Cliches',
        'Whataboutism'
    ]
    # Options:
    # - Classifier:
    #   - single: one classifier to 11 techniques
    #   - multi: 11 binary classifiers
    # - Data
    #   - One sentence at a time <-
    #   - One sentece plus some context (e.g. [CLS] <sentence_target> [SEP] <sentence_prev> <sentence_next>)
    #   - Fix window (n sentences up to 512 tokens)
    # - Flavours
    #   - IOB tagging
    #   - CRF
    # Window, or extend
    def __init__(self, split='train'):
        super(PTC, self).__init__()
        self.split = split
        self.df = self._get_dataframe(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        example = self.df.iloc[idx] # sentence, techniques_array(techinique, start, end)
        sentence, techniques = example['sentence'], example['spans']
        return sentence, techniques

    def _get_dataframe(self, split, overwrite=False):
        assert split in ['train', 'dev', 'test']
        path_articles = f'data/protechn_corpus_eval/{split}/'
        path_labels = f'data/protechn_corpus_eval/{split}/'
        article_paths = sorted(glob.glob(path_articles + '*.txt'))
        sentences = []
        span_labels = []
        article_ids = []
        offsets = []
        for path_article in article_paths:
            with open(path_article, 'r', encoding="utf8") as f_article:
                article_txt = f_article.read()
                article_id = path_article.split('/')[-1].replace('.txt', '').replace('article', '')
                labels_filename = 'article' + article_id + '.labels.tsv'
                with open(path_labels + labels_filename, 'r', encoding="utf8") as f_lables:
                    article_labels_txt = f_lables.read()
                    new_sentences, new_span_labels, new_offsets = self._split_in_sentences(article_id, article_txt, article_labels_txt)
                    sentences += new_sentences
                    span_labels += new_span_labels
                    article_ids += [article_id] * len(new_sentences)
                    offsets += new_offsets
        df = pd.DataFrame({'article_id': article_ids, 'sentence': sentences, 'spans': span_labels, 'offset': offsets})
        if split == 'train':
            df = self._consolidate(df)
        return df

    def _consolidate(self, df):
        df_unique = df[~df.duplicated(subset=['sentence'], keep=False)]
        df_duplicated = df[df.duplicated(subset=['sentence'], keep=False)]
        article_ids = []
        sentences = []
        spans = []
        offsets = []
        for example in df_duplicated.itertuples():
            if example.sentence not in sentences:
                df_same = df_duplicated[df_duplicated.sentence == example.sentence]
                _spans = [sp for _example_spans in df_same.spans.values for sp in _example_spans]
                map_techniques = {}
                _compacted = []
                for technique, pos in _spans:
                    map_techniques.setdefault(technique, [])
                    map_techniques.get(technique).append((pos))
                for key in map_techniques.keys():
                    mask = np.zeros(len(example.sentence), dtype=int)
                    for start, end in map_techniques.get(key):
                        mask[start:end] = 1
                    mask = list(mask)
                    index_changes = []
                    try:
                        value, index = 1, 0
                        while index < len(example.sentence):
                            index = mask.index(value, index)
                            index_changes.append(index)
                            value = int(not value) # Negate value
                    except ValueError:
                        pass
                    if len(index_changes) % 2 == 1:
                        index_changes.append(len(example.sentence))
                    for i in range(0, len(index_changes), 2):
                        _compacted.append((key, (index_changes[i], index_changes[i+1])))
                article_ids.append(example.article_id)
                sentences.append(example.sentence)
                spans.append(_compacted)
                offsets.append(example.offset)
        df_clean = pd.DataFrame({
            'article_id': article_ids,
            'sentence': sentences,
            'spans': spans,
            'offset': offsets
        })
        df = pd.concat([df_unique, df_clean])
        return df

    def _split_in_sentences(self, article_id, article_txt, article_labels_txt):
        label_lines = [line.split('\t') for line in article_labels_txt.split('\n') if line ]
        assert not label_lines or article_id == label_lines[0][0], f'{label_lines}; {article_id}'
        labels_in_article = [
            (
                self.LABELS.index(l[1]), # Propaganda technique index
                tuple(map(lambda a: int(a), l[-2:])) # (start, end) position
            ) for l in label_lines if l[1] in self.LABELS
        ]
        sentences, labels, offsets = self._do_split_in_sentences(article_txt, labels_in_article)
        return sentences, labels, offsets

    def _do_split_in_sentences(self, txt, labels_in_txt, split_by='\n'):
        """
        Returns:
        - txt splits
        - labels
        """
        txt_splits = txt.split(split_by)
        labels_in_txt = sorted(labels_in_txt, key=lambda a: a[1]) # sort by position
        output, labels, offset, offsets, split_len = [], [], 0, [], len(split_by)
        for txt_split in txt_splits:
            if not txt_split:
                # empty txt_splits
                offset += split_len
                continue
            labels_in_txt_split = []
            max_offset = offset + len(txt_split)
            for label, (start, end) in labels_in_txt:
                if start < max_offset and end > offset:
                    technique_start = max(0, start - offset)
                    technique_end = min(len(txt_split), end - offset)
                    assert technique_start < technique_end
                    labels_in_txt_split.append((label, (technique_start, technique_end)))
                elif start >= max_offset:
                    break
            offsets.append(offset)
            offset = max_offset + split_len
            labels.append(labels_in_txt_split)
            output.append(txt_split)
        return output, labels, offsets


class PTCLoadedLanguage(PTC):
    LABELS = ['Loaded_Language']


class PTCNameCallingLabeling(PTC):
    LABELS = ['Name_Calling,Labeling']


class MPTC(Dataset):
    """Meme Propaganda Techniques Corpus
    SemEval-2021 Task 6: Detection of Persuasive Techniques in Texts and Images
    https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus
    """

    LABELS = [
        'Appeal to authority',
        'Appeal to fear/prejudice',
        'Bandwagon',
        'Black-and-white Fallacy/Dictatorship',
        'Causal Oversimplification',
        'Doubt',
        'Exaggeration/Minimisation',
        'Glittering generalities (Virtue)',
        'Loaded Language',
        'Name calling/Labeling',
        "Misrepresentation of Someone's Position (Straw Man)",
        'Flag-waving',
        'Obfuscation, Intentional vagueness, Confusion',
        'Presenting Irrelevant Data (Red Herring)',
        'Reductio ad hitlerum',
        'Repetition',
        'Slogans',
        'Smears',
        'Thought-terminating cliché',
        'Whataboutism'
    ]
    def __init__(self, split='train'):
        super(MPTC, self).__init__()
        assert split in ('train', 'dev', 'test')
        self.split = split
        self.df = self._get_dataframe(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        example = self.df.iloc[idx] # sentence, techniques_array(techinique, start, end)
        sentence, labels = example['text'], example['labels']
        spans = [(self.LABELS.index(label.get('technique')), (label.get('start'), label.get('end')))
                 for label in labels if label.get('technique') in self.LABELS]
        return sentence, spans

    def _get_dataframe(self, split):
        split = split if split != 'train' else 'training'
        df = pd.read_json(f'data/SEMEVAL-2021-task6-corpus/{split}_set_task2.txt')
        # Tokenizers might use BPE or other strategy for out of vocabulary words (f.i. uppercase words).
        # This dataset contains many instances of uppercased sentences, so we true-case the text to allow for a greater
        # shared representation across tasks.
        df['text'] = df.text.apply(lambda s: util.get_true_case(s))
        return df


class MPTCLoadedLanguage(MPTC):
    LABELS = ['Loaded Language']


class MPTCNameCallingLabeling(MPTC):
    LABELS = ['Name calling/Labeling']

