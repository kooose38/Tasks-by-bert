## Proofreading with bert 
Generally, there are various tasks in grammar proofreading, but here we will perform the task of proofreading incorrect conversion of kanji. When selecting a kanji from conversion candidates, you accidentally select a kanji with a different meaning.

---

mistake: 努力が実り入省した。
correct: 努力が実り入賞した。

---

The task is to identify sentences due to such incorrect conversion from words and correct them.   

### Understanding file structure 

```files 
.
├── __init__.py
├── code
│   ├── __init__.py
│   ├── dataloader.py # 
│   ├── dataset # training dataset from wikipedia 
│   │   ├── test.txt
│   │   └── train.txt
│   ├── debug.py
│   ├── example_dataset.ipynb #Displaying the actual data format 
│   ├── model # Save the trained model with the lowest validation loss value 
│   ├── model.py # bertforMaskedLM by pytorch-lightning 
│   ├── predict.py # Infer with optimal weight 
│   ├── tokenizer.py # make a word a token for input bertmodel
│   └── trainer.py # do learning 
├── demo.py # please run this file 
└── tutorial.md # current files

```
### Make inferences 

Here, it is assumed that python can be executed. Please refer to other pages.  
  
1. First, let's install the required packages 

```command
 $ pip install -r requirement.txt
```

2. Execution of inference function  

When you're done, go to `demo.py` and run the following command: 
Now let's look at the arguments of the function. The best file for the trained model is set in bert_model_path.
For text, specify the sentence you want to infer.
If you want to try the model, it is recommended that you dare to enter sentences using kanji due to incorrect conversion.
And you can also specify the gpu, but in most cases it is not necessary.    

```command
  $ python3 demo.py 優勝トロフィーを返還しました
```
  
Was it displayed correctly? Thank you !

### Understanding learning 

First, take a look at [trainer.py](https://github.com/kooose38/Tasks-by-bert/blob/master/grammer/code/trainer.py). Here, training, verification, and reading of test data are performed. The score is detected from the sentences that will be predicted from the test data after learning. In the argument, specify **number of tokens**, **number of batches** for training data, and **learning rate**. The number of tokens increases in proportion to the time it takes to complete learning. Of course, you can also specify ** gpu **, so it is recommended to execute it in the environment. bert has many parameters. So it will take a lot of time to learn. Check the `./model/` file after learning. Hopefully the best model will be stored there. I hope you can do it:)

```command
  $ cd ./code  
  $ python3 trainer.py 32 32 0.01 
```