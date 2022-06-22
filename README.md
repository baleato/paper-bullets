# Datasets

- [Shared Task on Metaphor Detection](https://github.com/EducationalTestingService/metaphor/tree/master/VUA-shared-task)
  ```
  Leong, Chee Wee (Ben); Beigman Klebanov, Beata; Hamill, Chris; Stemle, Egon; Ubale, Rutuja; Chen, Xianyang; 2020
  A Report on the 2020 {VUA} and {TOEFL} Metaphor Detection Shared Task
  https://aclanthology.org/2020.figlang-1.3/
  ```

- [Propaganda Techniques Corpus](https://aclanthology.org/D19-1565/)
  ```
  Giovanni Da San Martino; Seunghak Yu; Alberto Barron-Cede; Rostislav Petrov; Preslav Nakov; 2019
  Fine-Grained Analysis of Propaganda in News Articles
  https://aclanthology.org/D19-1565/
  ```

- [SemEval-2021 Task6, subtask 2](https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus)
  ```
  Dimitar Dimitrov and Bin Ali, Bishr and Shaden Shaar and Firoj Alam and Fabrizio Silvestri and Hamed Firooz and Preslav Nakov and Da San Martino, Giovanni
  SemEval-2021 Task 6: Detection of Persuasion Techniques in Texts and Images
  https://aclanthology.org/2021.semeval-1.7/
  ```

## Setup

```
$ python3 -m venv venv
$ source venv/bin/activate
(venv)$ python -m pip install --upgrade pip
(venv)$ pip install -r requirements.txt
(venv)$ cd data
(venv)$ sh download.sh
...
(venv)$ cd ..
```

# Usage

```bash
usage: train.py [-h] [--config CONFIG] [--gpu GPU] [--pos_weight] [--folds FOLDS] [--kfold_seed KFOLD_SEED]
                [--encoder {bert-base-cased,bert-base-uncased,bert-large-cased,bert-large-uncased,roberta-base,roberta-large}] [--save_path SAVE_PATH]
                [--csv_results_file CSV_RESULTS_FILE] [--batch_size BATCH_SIZE] [--unfreeze_num UNFREEZE_NUM] [--dropout DROPOUT] [--load_model LOAD_MODEL]
                [--num_epochs NUM_EPOCHS] [--log_every LOG_EVERY] [--bert_lr BERT_LR] [--head_lr HEAD_LR] [--lr LR] [--seed SEED]
                [--training_tasks [{Metaphor,MetaphorSTAll,MetaphorSTVerbs,News,News_LoadedLanguage,News_NameCalling,Memes,Memes_LoadedLanguage,Memes_NameCalling} ...]]
                [--validation_task {Metaphor,MetaphorSTAll,MetaphorSTVerbs,News,News_LoadedLanguage,News_NameCalling,Memes,Memes_LoadedLanguage,Memes_NameCalling}]
                [--prefix PREFIX] [--patience PATIENCE] [--warmup WARMUP] [--lr_scheduler {constant,linear,cosine}] [--weight_decay WEIGHT_DECAY]
                [--task_sampling_ratio [TASK_SAMPLING_RATIO ...]] [--epoch_factor [EPOCH_FACTOR ...]] [--alignment_strategy {min1,max1,prop}]
                [--loss_task_factor [LOSS_TASK_FACTOR ...]] [--train_ignored_labels] [--no-train_ignored_labels]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config file path.
  --gpu GPU
  --pos_weight
  --folds FOLDS
  --kfold_seed KFOLD_SEED
  --encoder {bert-base-cased,bert-base-uncased,bert-large-cased,bert-large-uncased,roberta-base,roberta-large}
  --save_path SAVE_PATH
  --csv_results_file CSV_RESULTS_FILE
  --batch_size BATCH_SIZE
  --unfreeze_num UNFREEZE_NUM
  --dropout DROPOUT
  --load_model LOAD_MODEL
  --num_epochs NUM_EPOCHS
  --log_every LOG_EVERY
  --bert_lr BERT_LR
  --head_lr HEAD_LR
  --lr LR
  --seed SEED
  --training_tasks [{Metaphor,MetaphorSTAll,MetaphorSTVerbs,News,News_LoadedLanguage,News_NameCalling,Memes,Memes_LoadedLanguage,Memes_NameCalling} ...]
  --validation_task {Metaphor,MetaphorSTAll,MetaphorSTVerbs,News,News_LoadedLanguage,News_NameCalling,Memes,Memes_LoadedLanguage,Memes_NameCalling}
  --prefix PREFIX
  --patience PATIENCE
  --warmup WARMUP
  --lr_scheduler {constant,linear,cosine}
  --weight_decay WEIGHT_DECAY
  --task_sampling_ratio [TASK_SAMPLING_RATIO ...]
  --epoch_factor [EPOCH_FACTOR ...]
  --alignment_strategy {min1,max1,prop}
                        Used to calculate how many batches fit into the current epoch. - "prop" (proportional): the size of the epoch is calculated as the dot product between the
                        task sampling probability and the number of batches for that task. - "min1": limits the total number of batches per task in the epoch to the number of
                        batches of the smallest task. Sampling ratios are not taking into consideration. - "max1": limits the total number of batches per task in the epoch to the
                        number of batches of the smallest task. Sampling ratios are not taking into consideration.
  --loss_task_factor [LOSS_TASK_FACTOR ...]
  --train_ignored_labels
                        Whether to train the tokens the task flagged to ignore (e.g. non content words for Metaphor ALL-POS shared task). Enabled by default, use --no-
                        train_ignored_labels to turn off this setting.
  --no-train_ignored_labels
```

Example:

- Train a single task

```bash
$ python train.py --training_tasks News
```

- Multi-task training

```bash
$ python train.py --training_tasks MetaphorSTAll Memes --validation_task Memes --alignment_strategy min1
```
