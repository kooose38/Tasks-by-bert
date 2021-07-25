### Perform the task of filling in the text.
The sentences to be entered in this task are limited to Japanese and one-line sentences.  
We didn't do any additional training because we are leveraging a pre-trained model by bert. In other words, we will respond only by pre-learning.
Because it is learned by common sentences, it is vulnerable to words that are limited in a particular area.  
#### Understanding the file structure

```files
  .
├── README.md  # current position
├── __init__.py
├── code
│   ├── __init__.py
│   ├── confirm.py # do print some patterns 
│   ├── data.py # Randomly mask the input statement
│   ├── debug.py # 
│   ├── model.py # bert model 
│   ├── predict.py # It ’s the main method of making predictions.
│   └── tokenizer.py # create token 
└── demo.py # execution environment
```  

#### Make inferences  
1. Please install the required packages  
see [https://github.com/kooose38/Tasks-by-bert/tree/master/grammer](https://github.com/kooose38/Tasks-by-bert/tree/master/grammer)  
2. Execution of inference function  
Move to the hierarchy where `demo.py` exists.
Then execute the following command.
See the `-h` option for a description of the arguments. If this is your first time, this description will be useful.
I hope you work :) Thank you!

```command
  $ python3 demo.py -h 
  $ python3 demo.py オリンピックが開催された -hh beam 
```