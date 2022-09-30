import json
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from happytransformer import HappyQuestionAnswering, QATrainArgs
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import Trainer, TrainingArguments
import torch

# class Simple_Transformer_Trainer:
#     def __init__(self):
#         pass
#
#     def load_data(self, train_path, test_path):
#         with open(train_path, "r") as train_file:
#             train = json.load(train_file)
#
#         self.train_data = train
#
#         with open(test_path, "r") as test_file:
#             test = json.load(test_file)
#
#         self.test_data = test
#
#     def fine_tune(self, model_type, model_name, train_args):
#         model = QuestionAnsweringModel(model_type, model_name, args=train_args, use_cuda=False)
#         model.train_model(train_data=self.train_data, eval_data=self.test_data)
#         return model
#
#     def predict(self, model, input):
#         answers, probabilities = model.predict(input)
#         return answers[0]['answer']
#
#     def load_finetuned_model(self, model_name, model_path, use_cuda=False):
#         model = QuestionAnsweringModel(model_name, model_path, use_cuda=use_cuda)
#         return model




class Happy_Transformer_Trainer:
    def __init__(self):
        pass

    def prepare_single_example(self, example):
        answer = example["answers"]["text"][0]
        example["answer_start"] = example["answers"]["answer_start"][0]
        example["answer_text"] = answer
        return example

    def preprocess_data(self):
        raw_datasets = load_dataset("squad")
        raw_datasets = raw_datasets.remove_columns(["id", "title"])
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=40).select(range(10))
        raw_datasets['validation'] = raw_datasets['validation'].shuffle(seed=40).select(range(3))

        raw_datasets = raw_datasets.map(self.prepare_single_example)
        raw_datasets = raw_datasets.remove_columns("answers")
        df_train = raw_datasets['train'].to_pandas()
        df_train.to_csv('datasets/train.csv', index=False)
        df_validation = raw_datasets['validation'].to_pandas()
        df_validation.to_csv('datasets/validation.csv', index=False)



    def evaluate_model(self, model, test_set_filename):
        result = model.eval(test_set_filename)
        return result.loss

    def fine_tune(self, args, save_path):
        happy_qa = HappyQuestionAnswering()
        happy_qa.train("datasets/train.csv", args=args)
        happy_qa.save(save_path)


    def predict_text(self, model, context, question):
        result = model.answer_question(context, question)
        return result[0].answer

    def evaluate_model(self, model, test_set_filename):
        result = model.eval(test_set_filename)
        return result.loss

    def load_finetuned_model(self, path):
        model = HappyQuestionAnswering(load_path=path)
        return model



class HuggingFace_Transformer_Trainer:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        pass

    def prepare_single_example(self, example):
        answer = example["answers"]["text"][0]
        example["answer_start"] = example["answers"]["answer_start"][0]
        example["answer_end"] = example["answer_start"] + len(answer)
        return example

    def preprocess_data_qa(self):
        raw_datasets = load_dataset("squad")
        raw_datasets = raw_datasets.remove_columns(["id", "title"])
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=40).select(range(50))
        raw_datasets['validation'] = raw_datasets['validation'].shuffle(seed=40).select(range(10))
        raw_datasets = raw_datasets.map(self.prepare_single_example, remove_columns=["answers"])

        return raw_datasets

    def tokenize_data(self, raw_datasets):
        input_datasets = raw_datasets.map(self.preprocess_training_examples)
        filter_input_datasets = input_datasets.filter(
            lambda example: ((example['start_positions']) != 0) | (example['end_positions']) != 0)

        datasets = filter_input_datasets.remove_columns(
            column_names=['context', 'question', 'answer_start', 'answer_end'])

        train_data = datasets['train']
        validation_data = datasets['validation']

        train_data.set_format(type='pt')
        validation_data.set_format(type='pt')

        return train_data, validation_data

    def find_labels(self, offsets, answer_start, answer_end, sequence_ids):

        start_position = 0
        end_position = 0
        arr_idx = -1

        for i in range(offsets.shape[0]):

            # compute nr of tokens which belong to the question
            idx = 0
            while sequence_ids(i)[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids(i)[idx] == 1:
                idx += 1
            context_end = idx - 1

            for j in range(context_start, context_end):
                if offsets[i][j][0] == answer_start:
                    start_position = j

                if offsets[i][j][1] == answer_end:
                    end_position = j
                    arr_idx = i

        if start_position != 0 and end_position != 0 and arr_idx != -1:
            return start_position, end_position, arr_idx
        else:
            return (0, 0, -1)

    def preprocess_training_examples(self, example):
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        inputs = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            padding="max_length",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors='np'
        )

        start_position, end_position, arr_idx = self.find_labels(inputs['offset_mapping'], example["answer_start"],
                                                            example["answer_end"],
                                                            inputs.sequence_ids)

        return {'input_ids': inputs['input_ids'][arr_idx],
                'attention_mask': inputs['attention_mask'][arr_idx],
                'token_type_ids': inputs['token_type_ids'][arr_idx],
                'start_positions': start_position,
                'end_positions': end_position}

    def fine_tune(self, train_data, validation_data):
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint)

        training_args = TrainingArguments(
            output_dir='model_results',  # output directory
            overwrite_output_dir=True,  # overwrite the output_dir if it exists
            num_train_epochs=1,  # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=8,  # batch size for evaluation
            warmup_steps=20,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir=None,  # directory for storing logs
            logging_steps=50  # in each 50 training steps / after each 50 updates print the logging (loss and metric)
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_data,  # training dataset
            eval_dataset=validation_data,  # evaluation dataset
        )

        trainer.train()

        trainer.save_model(output_dir='transformers/mymodel')


    def predict_single_example(self, model_path, input):
        model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)

        inputs = self.tokenizer.encode_plus(input['question'], input['context'], return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_model = model(**inputs)
        answer_start = torch.argmax(answer_model['start_logits'])
        answer_end = torch.argmax(answer_model['end_logits']) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer

