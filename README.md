# log-quality-models

 <a href="https://logsight.ai/"><img src="https://logsight.ai/assets/images/logsight_logo.png" width="150"/></a>

Collection of models that can be used to assess the quality of log messages.

Used in
- [check-log-quality](https://github.com/aiops/check-log-quality)
- [logsight.ai](https://logsight.ai/)


## Usage

### Publish new model 

1.) Define the log quality assessment type `<type>`. It can be either `level` or `ling` (short for linguistic)

2.) Define a name for your model `<model_name>`. It should be descriptive. A unofficial convention is to use a arbitrary name, and the relevant components of the model separated by underscore. Do not use white spaces. Example: `qulog_sm_svc`.

3.) Call the prepare script: `prepare_<type> <model_name>`. It will create the directory structure for the model and prepare the template files.

4.) Copy your model file into `./<type>_quality/<model_name>/<model_name>`. The file needs to be named `model`.

5.) Adjust `./<type>_quality/<model_name>/setup.py`. Add all dependencies. Example:
```python
install_requires=[
    "scikit-learn==0.24.2",
    "spacy==3.1.0",
],
```
It's a python list. Don't forget the commas.

6.) Open `./<type>_quality/<model_name>/<model_name>/<model_name>.py`, implement the three methods, and adjust the imports. Do not change the things that are marked as not to be changed unless you really know what you are doing.
- `load`: Should load the `model` file and return a model object.
- `predict`: Should make a prediction for a single string (i.e. a log message).
- `predict_batch`: Should make a prediction for a list of strings (i.e. several log messages).

7.) Run `build_whl` without arguments. This will create a wheel file for the model.

8.) Commit, push, create a pull request, and ping [alek-thunder](https://github.com/alek-thunder).

### Update an existing model

1.) Check the path of the model you want to update. You will need the log quality assessment type `<type>` and the model name, which can be inferred from the path as follows: ``./<type>_quality/<model_name>`.

2.) Copy your new model into `./<type>_quality/<model_name>/<model_name>`. The model file must be named `model`.

2.) Run `build_whl -t <type> <model_name>`. This will recreate the wheel.

## Further remarks

You are free to add the respective training scripts for your models. Put them into `./training_scripts/<type>/<model_name>`. It should include a `train.py` file to train the model.
Do not push training data or model binaries. Link the training data via a README.md entry or a comment in `train.py` that tells where training data can be acquired.

