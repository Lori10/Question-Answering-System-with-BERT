# Question-Answering-NLP-System

## Table of Content
  * [Problem Statement](#Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Bert Tokenizer](#Bert-Tokenizer)
  * [Fine Tune BERT Transformer](#Fine-Tune-BERT-Transformer)
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
2. Using the tokenizer we get the tokenized input. The training data that the Transformer expects the following : features which are the input_ids and attention_mask of the tokenized input (question and context) and the target which is the index(position) of the token in the tokenized input where the answer starts and the index(position) of the token in the tokenized input where the answer ends. The data must be in a dictionary format where the keys are the input_ids, attention_mask, start_token_index and end_token_index. The methods find_labels and preprocess_training_examples is needed for this step. Since we will be fine-tuning BERT we must tokenizer our data using the BERT Tokenizer from HuggingFace. But how does the BERT Tokenizer tokenize/preprocess the data into a format that BERT model accepts ?! Lets explain it in more details in the next section.

## Bert Tokenizer
1. Apply Word Pience Tokenization to the raw sentence (can also be many sentences which is much more applicable for NLP tasks such as generating question and answer papers). Word Piece Tokenization is a technique that is used to segment sentences into tokens and is based on pre-trained Models with the dimension of 768 (max_length; for different BERT models we should check the max_length that the model accepts).
2. Add special tokens like : CLS (101) is a special token which comes in front of each input which many include many sentences and SEP (102) which is separater token (indicates the end of a sentence and beginning of another sentence).
3. Apply Token ID to determine token embedding/id for the individual tokens based on the vocabulary (dictionarty). In the vocabulary we map each token to a specific number (Note : the vocabulary is built on the training set and then used to encode validation and test set ). All I have to do is simply look at the tokens in the 768 dimension vector that I mentioned before. Here, the token CLS gets an embedding of 101 because that is the position of CLS in that 768 dimension. Similarly, the token love gets a token embedding of 2293, the token this gets an token embedding of 2023, and so on.
4. Segment embedding becomes much more important when there are multiple sentences in the input sequence. The segment ID of 0 represents that a sentence is the first sentence in the sequence, and similarly the segment embedding of 1 represents that it is a second sentence in the input sequence. Here I have only one sentence in my input sequence. So for all the individual tokens, I get a segment embedding of 0.
5. The position embedding determines the index portion of the individual token in the input sequence. Here, my input sequence consists of four tokens. So based on a zero based index, you can see the position embedding tokens for all the tokens. The position embedding for the token CLS is 0, the position embedding for the token love is 1, and so on.
6. Finally we sum the position, segment and token embedding that have been previously determined (sum of each dimension)
7. To have same length for all sequences we perform padding by adding 0s in the end so that new length=max_length (768). So the final embeddining of one training example is of shape (1, 4, 768). We should keep in mind that all training exampled will be encoded in this way.


All Cases using that we need to handle during preprocessing

* Some examples have start_index and end_index equal to -1 which indicates that there is no answer available (or this question is not answerable). In this case we can encode start_position and end_position to be 0 (CLS token index).
* Tokenized input length is lower than max_length. In this case we perform padding.
* Tokenized input length is greater than max_length and tokenized question length is lower than max_length (common case). In this case we perform truncation='only_second' to keep the question and truncate/discard tokens from the context. The Answer can be truncated or not depending on the context length and max_length. If the answer has been truncated we can either just discard it and set start_position and end_position to be equal to max_length to indicate that answer was truncated OR (better approach) to not discard the answer (these examples) we can perform special encoding with return_overflowing_tokens=True by encoding for a single example many sequences of max_length and allowing some overlapping use stride attribute. We feed to the model the tokenized sequence that contains the answer. This is a better approach since we do not discard the answer. But even in this case the answer can be splitted in different tokenized sequences; so in this case we return 0,0 for the start and end_position.
* Tokenized input length is greater than max_length and tokenized question length is greater than max_length. In this case it will be generated an error. We should either increase max_length or discard these examples if they are not too many.


## Fine Tune BERT Transformer

* BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives: 1) Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence. 2) Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.
This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the BERT model as inputs.
* In practice, fine-tuning is the most efficient way of applying pretrained transformers to new tasks, thereby reducing training time. That's why I am going to fine-tune the BERT Transformer using the SQUAD dataset.

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
