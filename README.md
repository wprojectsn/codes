# codes
Pytorch implementation for paper of "Concept Pointer Network for Abstractive Summarization".

#
pytorch 0.4 with python 2.7
#

##### How to train?
###### before train
*  Our conceptual vocabulary in code/data/vocabulary/concept_vocab,the number of conceptual words needs to be set before train.
*  You need to change some path and parameters in code/codespace/data_util/config.py before train.
###### Cross-Entropy Object Function
* python train.py None or python train.py (position of model parameters)
* When you use Cross-Entropy Object Function train the model,set RL_train = False and DS_train = False in code/codespace/data_util/config.py
###### reinforcement learning(RL)
* Before use RL train the model, you need use Cross-Entropy Object Function train the model and set the Cross-Entropy train times in code/codespace/data_util/config.py, finally the model will automatically use reinforcement learning to train the model when the train times exceed the Cross-Entropy train times.
* When you use RL train the model,set RL_train = True and DS_train = False in code/codespace/data_util/config.py
###### Distant Supervision(DS)
* Before use DS train the model, you need use Cross-Entropy Object Function train the model and Retain model parameters,then use the command “python train.py (position of model parameters)” train the model.
* When you use DS train the model,set DS_train = True and RL_train = False in code/codespace/data_util/config.py
##### How to test?
* python decode.py (position of model parameters)
#
##### other
* Our concepts come from https://concept.research.microsoft.com/Home/API
* Part of our code references https://github.com/atulkum/pointer_summarizer
