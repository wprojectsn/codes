# codes
Pytorch implementation for paper of "Concept Pointer Network for Abstractive Summarization".

#
pytorch 0.4 with python 2.7
#

##### How to train?
* python train.py None or python train.py (the model name)
###### before train
* 1 Our conceptual vocabulary in code/data/vocabulary/concept_vocab,the number of conceptual words needs to be determined before train.
* 2 you need to change some path and parameters in code/codespace/data_util/config.py
##### How to test?
* python decode.py (the model name)

##### other
* Our concepts come from https://concept.research.microsoft.com/Home/API
* Some of our code references https://github.com/atulkum/pointer_summarizer
