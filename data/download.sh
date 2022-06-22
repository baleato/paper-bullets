#!/bin/bash

# TODO:
# - Download propaganda datasets instead of including them in the repo

mkdir -p VUA/2541
curl -Lo VUA/2541/VUAMC.xml 'https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2541/VUAMC.xml'
# Steps from Shared Task on Metaphor Detection
# https://github.com/EducationalTestingService/metaphor/tree/master/VUA-shared-task
curl -Lo VUA/naacl_flp_starter_kit.zip 'https://github.com/EducationalTestingService/metaphor/releases/download/v1.0/naacl_flp_starter_kit.zip'
unzip VUA/naacl_flp_starter_kit.zip -d VUA/
curl -Lo VUA/naacl_flp_testing_kit.zip 'https://github.com/EducationalTestingService/metaphor/releases/download/v1.0/naacl_flp_testing_kit.zip'
unzip VUA/naacl_flp_testing_kit.zip -d VUA/
pushd VUA
# Generates VUA/vuamc_corpus_train.csv (words)
# requires python module lxml
python3 vua_xml_parser.py
# Generates VUA/vuamc_corpus_test.csv
python3 vua_xml_parser_test.py
popd

# Labels to report f-measure (1st VUA Verbs: 0.804; 1st VUA ALLPOS 0.769; results: https://competitions.codalab.org/competitions/22188#results)
curl -Lo VUA/naacl_flp_train_gold_labels.zip 'https://github.com/EducationalTestingService/metaphor/releases/download/v1.0/naacl_flp_train_gold_labels.zip'
unzip VUA/naacl_flp_train_gold_labels.zip -d VUA/naacl_flp_train_gold_labels
curl -Lo VUA/naacl_flp_test_gold_labels.zip 'https://github.com/EducationalTestingService/metaphor/releases/download/v1.0/naacl_flp_test_gold_labels.zip'
unzip VUA/naacl_flp_test_gold_labels.zip -d VUA/naacl_flp_test_gold_labels
