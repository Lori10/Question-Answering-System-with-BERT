# Question-Answering-NLP-System

## Table of Content
  * [Problem Statement](#Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Model Building](#Model-Building)
  * [Model Tuning](#Model-Tuning)
  * [Techniques to improve the model performance](#Techniques-to-improve-the-model-performance)
  * [Important Notes](#Important-Notes)
  * [Disadvantages of N gram language model and faced issues](#Disadvantages-of-N-gram-language-model-and-faced-issues)
  * [Demo](#demo)
  * [Bug and Feature Request](#Bug-and-Feature-Request)
  * [Future scope of project](#future-scope)


## Business Problem Statement
Question Answering (QnA) model is one of the very basic systems of Natural Language Processing. In QnA, the Machine Learning based system generates answers from the knowledge base or text paragraphs for the questions posed as input. Various machine learning methods can be implemented to build Question Answering systems. Using Natural Language Processing, we can achieve this objective. NLP helps the system to identify and understand the meaning of any sentences with proper contexts. In this project I am going to fine-tune the BERT Transformer which will take a context and a question as input, process the context and prepare the answer from it.

## Data
Data Source : Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset on HuggingFace, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

## Used Libraries and Resources
**Python Version** : 3.9

**Libraries** : transformers (huggingface), happytransformer

**References** : https://towardsdatascience.com/, https://machinelearningmastery.com/


## Data Preprocessing
I Preprocessed this data with the following steps:

1. The features are : 1. the question (string), the context(string). The target/label should be the index(position) where the answers in the context string starts and the index(position) in the context string where the answer ends. We assume that the answer is a substring of context string. If the data is not in this kind of format we should preprocess it to this type of features and labels (dictionary). The data must be in a dictionary format where the keys are the question, context, start_index, end_index. The method prepare_data is needed for this step. Note : In the dataset, we are given the starting index and the answer. We assume that the answer is substring of context string.
2. Using the tokenizer we get the tokenized input. The training data that the Transformer expects the following : features which are the input_ids and attention_mask of the tokenized input (question and context) and the target which is the index(position) of the token in the tokenized input where the answer starts and the index(position) of the token in the tokenized input where the answer ends. The data must be in a dictionary format where the keys are the input_ids, attention_mask, start_token_index and end_token_index. The methods find_labels and preprocess_training_examples is needed for this step.

All Cases using that we need to handle during preprocessing

* Some examples have start_index and end_index equal to -1 which indicates that there is no answer available (or this question is not answerable). In this case we can encode start_position and end_position to be 0 (CLS token index).
* Tokenized input length is lower than max_length. In this case we perform padding.
* Tokenized input length is greater than max_length and tokenized question length is lower than max_length (common case). In this case we perform truncation='only_second' to keep the question and truncate/discard tokens from the context. The Answer can be truncated or not depending on the context length and max_length. If the answer has been truncated we can either just discard it and set start_position and end_position to be equal to max_length to indicate that answer was truncated OR (better approach) to not discard the answer (these examples) we can perform special encoding with return_overflowing_tokens=True by encoding for a single example many sequences of max_length and allowing some overlapping use stride attribute. We feed to the model the tokenized sequence that contains the answer. This is a better approach since we do not discard the answer. But even in this case the answer can be splitted in different tokenized sequences; so in this case we return 0,0 for the start and end_position.
* Tokenized input length is greater than max_length and tokenized question length is greater than max_length. In this case it will be generated an error. We should either increase max_length or discard these examples if they are not too many.


## Model Building

* A n-gram language model assumes that the probability of the next word depends only on previous n-1 grams. (The previous n-1 gram is the series of the previous 'n-1' words/tokens). In case of bigram language model the probability of the next word depends on the previous gram/word.
* The conditional probability for the word at position 't' in the sentence, given that the words preceding it is :
![alt text](https://github.com/Lori10/Statistical-Grammer-Checker-FromScratch/blob/main/img1.PNG "Image")
* We can estimate this probability  by counting the occurrences of these series of words in the training data. The probability can be estimated as a ratio, where the numerator is the number of times word 't' appears after words t-1 till t-n in the training data/corpus and the denominator is the number of times word t-1 till t-n appears in the training data/corpus. In the bigram model, the numerator would be the number of times word 't' appears after words t-1 in the training data/corpus and the denominator is the number of times word t-1 appears in the training data/corpus. The function C() denotes the number of occurence of the given sequence. 
* This equation tells us that to estimate probabilities based on n-grams, we need the counts of n-grams (for denominator) and (n-1)-grams (for numerator). For example if we want to use trigram we need to calculate bigram and trigram counts. In case of using bigram model, we have to calculate bigram und unigram counts.
* When computing the counts for n-grams,  we prepare the sentence beforehand by prepending n-1 starting markers "<s\>" to indicate the beginning of the sentence and 1 ending marker "<e\>" to indicate the end of the sentence. This modification is done because we also want to calculate the probability of beginnig and ending the sentence given a particular word. Only when calculating bigram counts, we should prepend 2 "<s\>" markers and not 1 in order to include counting the occurance of the pair ("<s\>", "<s\>"). This is done because in case of using a trigram language model we would have in the beginnig of each sentence 2 markers "<s\>" "<s\>". When calculating the conditional probabilities for the word that are placed in the beginning of the sentence we need to know the occurance of ("<s\>", "<s\>") which is not found in the bigram counts since there we used only 1 "<s\>" marker. So, for example in case of trigram analysis if the tokenized sentence is ["I" , "like", "food"], modify it to be ["<s\>" , "<s\>", "I", "like", "food"]. We also must prepare the sentence for counting by appending an end token "<e\>" so that the model can predict when to finish a sentence. 
* We will store the counts as a dictionary. The key of each key-value pair in the dictionary is a **tuple** of n words (and not a list). The value in the key-value pair is the number of occurrences.  - The reason for using a tuple as a key instead of a list is because a list in Python is a mutable object (it can be changed after it is first created).  A tuple is "immutable", so it cannot be altered after it is first created.  This makes a tuple suitable as a data type for the key in a dictionary.
* The formula for the conditional probability doesn't work when a count of an n-gram is zero. Either the nominator may become 0 in case of having at least one conditional probability equal to 0. We should avoid these cases since we may have a sentence where the other words have very high probability of being close to each other (high conditional probabilities) and one single word (its conditional probability) may lead to a probability of a sentence to be 0 (We will see in the next step when calculating the probability of a sentence). Or the denominator may become 0 if $$w_{t-1}\dots w_{t-n}$$ does not appear in the training data/corpus. A way to handle zero counts is to use k-smoothing technique by adding a positive constant k (value between 0 and 1) to each numerator and k * |V| where |V| is the size of words in the vocabulary. For n-grams that have a zero count, the equation becomes k / k * |V|. The formula for the conditional probability becomes :
![alt text](https://github.com/Lori10/Statistical-Grammer-Checker-Corrector/blob/main/img3.PNG "Image")
* The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words. The real probability of a sentence is calculated using the chain rule. The probability of each word is calculated based on the entire previous words in that sentence. P(word_A, word_B, word_C, word_D) = P(word_A) * P(word_B/word_A) * P(word_C/word_A,word_B) * P(word_D/word_A,word_B,word_C). What we do instead using n-gram analysis, we take into consideration only the last n-1 words to calculate the probability of the next word because in natural language a word may be only depends on some previous words and not only on the entire previous words of the sentence. In our case we are going to detect basic grammer errors which can easily be detected by only check the 2 consecutive words. That's why we chose to use bigram language model. The estimated probability of bigram model would be : estimated_P(word_A, word_B, word_C, word_D) = P(word_A) * P(word_B/word_A) * P(word_C/word_B) * P(word_D/word_C).
* To evaluate the model we can use different metrics like auc score, f1-score, confusion matrix (False Positive, False Negative, True Positive, True Negative), accuracy score etc. In our case we want the model to detect the grammer errors which means that we should focus more in reaching a high True Positive Rate (Recall) / Low False Negative Rate. 


## Model Tuning
* We are going to perform hyperparameter tuning. Our hyperparameters for the bigram language model are the threshold and k-smoothing parameter. We are going to tune different values for threshold and k-smoothing parameter, check for each combination of them how the model behaves (measure different performance metrics like auc score, recall, f1 score, false positives) and choose the threshold that gives us the best performance. The metric that will determine the best model will be f1_score of class 0 since we want to focus more on predicting well sentences which are grammatically incorrect.
* The table below shows the performance metrics auc score, f1 score and recall for the default and tuned bigram language model.

| Model Name                        | AUC Score                | F1-Score                    |                 Recall |  
|:---------------------------------:|:------------------------:|:---------------------------:|:----------------------:|
|Default bigram language model      |     0.666                |     0.67                    |         0.67           |    
|Tuned bigram language model        |     0.833                |     0.857                   |                  1     |     

* After tuning the threshold and k smoothing parameter we could achieve a higher f1_score, higher auc score and higher recall which means an overall better model performance.

## Techniques to improve the model performance

1. Collecting more text data. We calculate the probability of new english sentences and it may be grammatically correct. But in case of having a small corpus (not huge enough) many words won't match any word from our corpus which will result in having a low probability and maybe classifying it as grammtically incorrect. By increasing the size of our corpus we increase the probabilty of seeing those words from the new test sentences in our corpus and as a result we'll have a more accuracte probability of whether a sentence can be grammatically correct or not.
2. Quality of the corpus/text data. In order to achieve good results in detecting grammer errors we assume that our corpus is grammtically correct. If there were grammer errors in our corpus, new sentences which are grammatically incorrect would have many matches in the corpus which would cause their probabilities to be high and as a result to be classified as grammatically correct.
2. Another very important parameter that affects our bigram model performance is the threshold. Threshold determines the point/the value where we classify a sentence as grammtically correct or not. So if we choose a relatively low threshold, it may lead to too many False Positives which we would not want. In case of using a relatively high threshold, it may lead to too many False Negatives which we definitely want to avoid. Since we care more about the False Negative Rate (detecting the sentences which are grammatically incorrect) maybe we could choose a bit lower threshold. One way to find a 'good' threshold would be to tune it. We can select a range of values and make predictions using different threshold (which means different models) on new test sentences and check how the language model performs. In the end we choose the value of threshold which gives the highest performance. Some good performance metrics that we might use to evaluate our different language models in our case would be F1 Score or Recall.
4. One other hyperparameter that affects our bi-gram language model performance is the k-smoothing parameter. There are different ways to perform Smoothing in language models for example : 
Add-One Smoothing, K-Smoothing etc. In our case we are going to apply K-Smoothing. The advantage of K-Smoothing consists of improving the probability distributions. In case of K=1 smoothing may lead to sharp changes in probabilities. For example two different sentences that have the same probability (without k-smoothing which means k=0), after applying k=1 smoothing they may have different probabilities.
5. Increasing the test set. The bigger the test set, the better our model can generalize on new test sentences. I used a special test set that includes some grammer errors like Noun-Verb agreement and Determinant-Noun agreement. It is still a very small test set and it would be a good idea to have such more examples and check how the model performs. We should keep in mind that the tuned threshold and k-smoothing parameter may overfit to this small test set.
6. To reach higher model performance we could use some other techniques for example : other smoothing techniques, interpolation, backoff which help us better estimate the probability of unseen n-gram sequences.

## Important Notes 
* We dont divide our corpus/text dataset into training and test set since we assume that the entire text data is grammatically correct. Since we want to catch specific grammer errors I will build the test set manually and check the language model performance especially on those chosen test sentences. We should always keep in mind that there is no guarentee that the model will perform well on other unseen test sets because our test is quite small.
* The bigram language model can detect grammer errors that include 2 grams/tokens for example noun-verb agreement, determinant-noun agreement, adjective order etc which we can detect using a bigram language model. If we want to catch other grammer errors on the long term we have to look at the words beyond 2 grams; thats why in those cases we should use n gram language model where n>=3.



## Disadvantages of N gram language model and faced issues
* The main disadvantage of n-gram language model is that it requires a lot of space and RAM. Especially in case of having long sentences the model should store the probabilities of all possible combinations and also all the n-gram-counts dictionaries. <br />
<b>Possible Solution</b> : Train a more advanced model like RNN etc. 
* N gram language model estimated the probability of a word given some previous words. In fact to estimate the probability of a word we should look at the previous and text words to capture the full context. <br />
<b>Possible Solution</b> : Use bidirectional RNN.
* Another disadvantage of n-gram language model : The longer the sentences the lower the probability becomes. Since we multiply by numbers that are lower than 1 the sentence probability decreases. This means that the longer the sentence the lower the probability that it is correct. The sentence may be very long and grammatically correct but is classified as grammatically incorrect by our model because of its high length. So it becomes very difficult to estimate the probability of long sentences due to their length. As a result the n-gram language model fails to capute long dependencies between words in a sentence <br />
<b>Possible Solution</b> : Finding out the right value of k-smoothing parameter since it affects the distribution of the probabilities or use other models like RNN.
* Basic sentences that are very commonly used are classified correctly , some sentences are grammatically correct but classified as grammatically incorrect by our bigram model since most of their words do not appear in our corpus. <br />
<b>Possible Solution</b> : increase the corpus size.

## Demo

This is how the web application looks like : 
We should input the sentence that we want to check for grammer errors and select the value of n for the n-gram language model. 

![alt text](https://github.com/Lori10/Statistical-Grammer-Checker-FromScratch/blob/main/demo.png "Image")


## Bug and Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Lori10/Statistical-Grammer-Checker-FromScratch/issues) here by including your search query and the expected result

## Future Scope
* Try other techniques to reach higher model performance for example : other smoothing techniques, interpolation, backoff which help us better estimate the probabilities of unseen n-gram sequences.
* Use other performance metrics to select the best model like perplexity etc.
* Optimize Flask app.py Front End.
