# Question-Answering-NLP-System

## Table of Content
  * [Problem Statement](#Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Bert Tokenizer](#Bert-Tokenizer)
  * [Fine Tune BERT Transformer](#Fine-Tune-BERT-Transformer)
  * [Make Predictions](#Make-Predictions)
  * [Demo](#demo)
  * [Bug and Feature Request](#Bug-and-Feature-Request)
  * [Future scope of project](#future-scope)


## Problem Statement
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
2. Using the tokenizer we get the tokenized input. The training data that the Transformer expects the following : features which are the input_ids and attention_mask of the tokenized input (question and context) and the target which is the index(position) of the token in the tokenized input where the answer starts and the index(position) of the token in the tokenized input where the answer ends. The data must be in a dictionary format where the keys are the input_ids, attention_mask, start_token_index and end_token_index. The methods find_labels and preprocess_training_examples is needed for this step. Since we will be fine-tuning BERT we must tokenizer our data using the BERT Tokenizer from HuggingFace. But how does the BERT Tokenizer tokenize/preprocess the data into a format that BERT model accepts ?! Lets explain it in more details in the next section (Bert Tokenizer Section).

All Cases using that we need to handle during preprocessing

* Some examples have start_index and end_index equal to -1 which indicates that there is no answer available (or this question is not answerable). In this case we can encode start_position and end_position to be 0 (CLS token index).
* Tokenized input length is lower than max_length. In this case we perform padding.
* Tokenized input length is greater than max_length and tokenized question length is lower than max_length (common case). In this case we perform truncation='only_second' to keep the question and truncate/discard tokens from the context. The Answer can be truncated or not depending on the context length and max_length. If the answer has been truncated we can either just discard it and set start_position and end_position to be equal to max_length to indicate that answer was truncated OR (better approach) to not discard the answer (these examples) we can perform special encoding with return_overflowing_tokens=True by encoding for a single example many sequences of max_length and allowing some overlapping use stride attribute. We feed to the model the tokenized sequence that contains the answer. This is a better approach since we do not discard the answer. But even in this case the answer can be splitted in different tokenized sequences; so in this case we return 0,0 for the start and end_position.
* Tokenized input length is greater than max_length and tokenized question length is greater than max_length. In this case it will be generated an error. We should either increase max_length or discard these examples if they are not too many.

## Bert Tokenizer
1. Apply Word Pience Tokenization to the raw sentence (can also be many sentences which is much more applicable for NLP tasks such as generating question and answer papers). Word Piece Tokenization is a technique that is used to segment sentences into tokens and is based on pre-trained Models with the dimension of 768 (max_length; for different BERT models we should check the max_length that the model accepts).
2. Add special tokens like : CLS (101) is a special token which comes in front of each input which many include many sentences and SEP (102) which is separater token (indicates the end of a sentence and beginning of another sentence).
3. Apply Token ID to determine token embedding/id for the individual tokens based on the vocabulary (dictionarty). In the vocabulary we map each token to a specific number (Note : the vocabulary is built on the training set and then used to encode validation and test set ). All I have to do is simply look at the tokens in the 768 dimension vector that I mentioned before. Here, the token CLS gets an embedding of 101 because that is the position of CLS in that 768 dimension. Similarly, the token love gets a token embedding of 2293, the token this gets an token embedding of 2023, and so on.
4. Segment embedding becomes much more important when there are multiple sentences in the input sequence. The segment ID of 0 represents that a sentence is the first sentence in the sequence, and similarly the segment embedding of 1 represents that it is a second sentence in the input sequence. Here I have only one sentence in my input sequence. So for all the individual tokens, I get a segment embedding of 0.
5. The position embedding determines the index portion of the individual token in the input sequence. Here, my input sequence consists of four tokens. So based on a zero based index, you can see the position embedding tokens for all the tokens. The position embedding for the token CLS is 0, the position embedding for the token love is 1, and so on.
6. Finally we sum the position, segment and token embedding that have been previously determined (sum of each dimension)
7. To have same length for all sequences we perform padding by adding 0s in the end so that new length=max_length (768). So the final embeddining of one training example is of shape (1, 4, 768). We should keep in mind that all training exampled will be encoded in this way.


## Fine Tune BERT Transformer

* BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives: 1) Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence. 2) Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.
This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the BERT model as inputs.
* In practice, fine-tuning is the most efficient way of applying pretrained transformers to new tasks, thereby reducing training time. That's why I am going to fine-tune the BERT Transformer using the SQUAD dataset.

## Make Predictions

1. First we tokenize the question and context using the same Tokenizer. We do not need to perform padding to avoid PAD tokens in the answer that we want to generate. But we have to make sure that the tokenized input length is lower or equal to model_max_length. If it is not the case, we have to truncate the sequence.
2. We feed the tokenized input to our model and generate the logits. The model generates the start_logits (for the start position/index) and end_logits (for end position/index) which are arrays of shape nr_examples x max_length. This means for the start position the model computes a probability (logit) of each token in the tokenized sequence of being the start_position.
3. To compute the final start and end index (postion) we can follow 2 approaches. We can compute the argmax of start_logits and end_logits (independently) to get the start and end position of each example OR we convert logits into probailities. Then we compute a score for each pair of start and end position by taking the product of start and end_probability. Finally we take start and end position of highest score. This is the approach that the Pipeline in HuggingFace use and its obviously a better technique to generate the correct answers since we take into consideration the best start and end index pair and not the best start and the best end index.
4. Using the start and end position we select the tokenized answer (subset from tokenized input) and convert tokens their corresponding strings to get the final answer.


## Demo

This is how the web application looks like : 
We should input the sentence that we want to check for grammer errors and select the value of n for the n-gram language model. 

![alt text](https://github.com/Lori10/Question-Answering-NLP-System/blob/main/img_qa.png "Image")


## Bug and Feature Request
If you find a bug, kindly open an issue.

## Future Scope
* Optimize Flask app.py Front End.
