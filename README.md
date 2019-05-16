# codes
Pytorch implementation for paper of "Concept Pointer Network for Abstractive Summarization".

#
pytorch 0.4 with python 2.7
#

##### How to train?
* python train.py None or python train.py (the model name)
###### before train
* 1 Our conceptual vocabulary in code/data/vocabulary/concept_vocab,the number of conceptual words needs to be set before train.
* 2 You need to change some path and parameters in code/codespace/data_util/config.py before train.
###### RL train
* Before use RL train the model, you need use Cross-Entropy Object Function train the model and set the Cross-Entropy train times in code/codespace/data_util/config.py, finally the model will automatically use reinforcement learning to train the model when the train times exceed the Cross-Entropy train times.
* when you use RL train the model,set RL_train = True and DS_train = False in code/codespace/data_util/config.py
###### DS train
* Before use DS train the model, you need use Cross-Entropy Object Function train the model and Retain model parameters,then set DStrain = True and RL_tran = False in code/codespace/data_util/config.py, finally use the command “python train.py (the model name)” train the model.
* when you use DS train the model,set DS_train = True and RL_train = False in code/codespace/data_util/config.py
##### How to test?
* python decode.py (the model name)
#
##### other
* Our concepts come from https://concept.research.microsoft.com/Home/API
* Some of our code references https://github.com/atulkum/pointer_summarizer
