from flask import Flask, request, render_template
from flask_cors import cross_origin
from flask import Response
#from trainModel import Simple_Transformer_Trainer
#from simpletransformers.question_answering import QuestionAnsweringModel
from trainModel import Happy_Transformer_Trainer
from happytransformer import HappyQuestionAnswering, QATrainArgs
from trainModel import HuggingFace_Transformer_Trainer


app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/finetune_happytransformer", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.method == "POST":
            happy_transformer = Happy_Transformer_Trainer()
            happy_transformer.preprocess_data()

            args = QATrainArgs(num_train_epochs=1)
            happy_transformer.fine_tune(save_path='FineTunedModel/', args=args)


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")



@app.route("/finetune_huggingface_transformer", methods=['POST'])
@cross_origin()
def FineTuneHugTrans():
    try:
        if request.method == "POST":
            model = HuggingFace_Transformer_Trainer('bert-base-cased')
            data = model.preprocess_data_qa()
            train_data, validation_data = model.tokenize_data(data)
            model.fine_tune(train_data, validation_data)

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predictRouteClient():
    try:
        if request.method == "POST":
            question = request.form['question']
            context = request.form['context']

            model_type = request.form['model_type']
            if model_type == 'HappyTransformer':
                happy_transformer = Happy_Transformer_Trainer()
                model = happy_transformer.load_finetuned_model(path='FineTunedModel/')
                answer = happy_transformer.predict_text(model, context, question)

                return render_template('home.html',
                                       prediction_text=f'Answer : {answer}')

            else:
                input = {'question': question,
                         'context': context}

                model = HuggingFace_Transformer_Trainer('bert-base-cased')
                answer = model.predict_single_example(model_path='transformers/mymodel', input=input)

                return render_template('home.html',
                                       prediction_text=f'Answer : ' + answer)

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Prediction successfull!!")



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
