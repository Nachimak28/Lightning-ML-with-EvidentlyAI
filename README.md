# Lightning ML with EvidentlyAI app


This ⚡ [Lightning app](lightning.ai) ⚡ was generated automatically with:

```bash
lightning init app data_centric_ml
# Note: I've changed the name of the github repo but this shall work regardless
```

## Demo output

This app is basically a demo to show a UI based ML workflow. 
**Steps:**
* The user uploads files for their train and test sets (CSV files for tabular data problem statements)
* User chooses a type of task - Classification or Regression
* User inputs the label/target column name andd submits files for processing.

Demo video showing the process:



https://user-images.githubusercontent.com/23210132/182327240-33cadaff-3a6c-4aae-bf81-45cf49f7a5aa.mp4



**What happens under the hood ?**
* The app ingests the files and generates an EvidentlyAI based Data analysis dashboard to understand the data drift and general feature analytics plotted in an easily interpretable way
* It simultaneously trains a simple RandomForest classifier/regressor based on the task type chosen
* Once the model training is complete, it generates the models predictions on the train and test set and gives insights in the model's performance metrics which are again plotted nicely using EvidentlyAI

**What about model serving?**
It is possible but given the dynamic nature of this app, its best to leave it to the admin to generate the serving related dashboard/UI. 

**What assumptions does this app make about the data?**
* The data has been preprocessed already - missing values amputation, scaling etc
* The app is tested using good quality data and no scaling is applied since the models trained are only RandomForests (scale invariant)

**What could have been done better?**
* The user has to wait and refresh the screens to get the outputs, more realtime ness would have been awesome but I'm still discovering this framework and need to understand the core in a better way
* More generalization and customization for more models support and maybe some standard preprocessing via the UI itself
* Dynamic UI (Streamlit/Gradio) generation for model serving based on the input data feature columns and their data types


## To run data_centric_ml

First, install data_centric_ml (warning: this app has not been officially approved on the lightning gallery):

```bash
lightning install app https://github.com/Nachimak28/Lightning-ML-with-EvidentlyAI
```

Once the app is installed, run it locally with:

```bash
lightning run app data_centric_ml/app.py
```

Run it on the [lightning cloud](lightning.ai) with:

```bash
lightning run app data_centric_ml/app.py --cloud
```

## to test and link

Pending tests. 
Run flake to make sure all your styling is consistent (it keeps your team from going insane)

```bash
flake8 .
```

To test, follow the README.md instructions in the tests folder.


## TODO

- [ ] Complete readme with relevant codebase and examples
- [ ] Refactor code to be more generalized and clean
- [ ] Write tests
