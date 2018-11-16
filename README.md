## Low-resourced Language Modeling based on Kaldi

This repository provides Kaldi users with a few useful scripts for **language modeling**, especially for speech recognition under **low-resourced conditions**.

![image](https://github.com/charlesliucn/kaldi-lm/misc/logo/framework.png)

The scripts are mainly based on `babel/s5d` in `./egs` directory.

Most of the scripts are in `babel/s5d` and `wsj/s5/steps` directories.

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
	- Word Vectors **Pre-training**
	- LSTM Language Model for **Rescoring**

* * *

### Contact
Any questions please send e-mails to charlesliutop@gmail.com.

* * *

More info about Kaldi Speech Recognition Toolkit, please see [Kaldi's official github repository](http://www.github.com/kaldi-asr/kaldi).
