## Low-resourced Language Modeling based on Kaldi

This repository provides Kaldi users with a few useful scripts for **language modeling**, especially for **low-resourced conditions**.
The scripts are mainly based on `babel/s5d` in `egs` directory.

Most of the scripts are in `babel/s5d` and `wsj/s5/steps`.

>>> Currently, the scripts are not so well organized. A document of detailed usage of these scripts will be added later.

![image](http://nbviewer.jupyter.org/github/charlesliucn/LanMIT/blob/master/misc/logo/framework.png)

* * *

### Main Contributions
+ **Data Augmentation**
	- Text Preprocessing for **Lexicon Generation**
	- **Vocabulary Expansion** Based on Word Frequency
	- **Data Selection** Based on Multiple Criteria
+ N-Gram Language Models based on SRILM
	- **Linear Interpolation** for N-Gram models
	- N-Gram Language Model for **Rescoring**
+ LSTM Language Model Based on Tensorflow
	- Word Vectors **Pre-training** for RNN/LSTM Language Model Training
	- LSTM Language Model for **Rescoring**


* * *

### Relevant Toolkits
+ [XenC](https://github.com/antho-rousseau/XenC): an open-source tool for data selection in Natural Language Processing.
+ [GloVe](https://github.com/stanfordnlp/GloVe): Global Vectors for Word Representation.
+ [SRILM](http://www.speech.sri.com/projects/srilm/): an Extensible Language Modeling Toolkit.

* * *

### Contact
Any questions please send e-mails to charlesliutop@gmail.com.

* * *

More info about Kaldi Speech Recognition Toolkit, please see [Kaldi's official github repository](http://www.github.com/kaldi-asr/kaldi).
