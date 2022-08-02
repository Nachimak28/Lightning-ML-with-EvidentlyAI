import os
import lightning as L
from lightning_app.components.serve import ServeGradio
from evidently_data_analysis import EvidentlyDataAnalysis
from evidently_model_analysis import EvidentlyModelAnalysis
import gradio as gr
from data_centric_ml import StaticPageViewer, ModelTrainer
from lightning.app.storage.path import Path
import tempfile
from shutil import copy2

class LitGradio(ServeGradio):

    inputs = [
                gr.inputs.File(label='Upload Train dataset'), 
                gr.inputs.File(label='Upload Test dataset'), 
                gr.inputs.Radio(["classification", "regression"], label='Select Task Type'),
                gr.inputs.Textbox(label='Enter target column name')
            ]
    outputs = gr.outputs.Textbox(label='output')
    examples = [
        ['resources/ba_cancer_train.csv', 'resources/ba_cancer_test.csv', 'classification', 'target'],
        ['resources/ca_housing_train.csv', 'resource/ca_housing_test.csv', 'regression', 'MedHouseVal']
        ]

    def __init__(self):
        super().__init__(parallel=True)
        self.data_dir = Path(tempfile.mkdtemp())
        self.train_file_path = None
        self.test_file_path = None
        self.task_type = None
        self.target_column_name = None

    def predict(self, train_file, test_file, task_type, target_column_name):
        # set paths of files to class variables for other components to use
        # get file names
        train_file_name = os.path.split(train_file.name)[-1]
        test_file_name = os.path.split(test_file.name)[-1]
        # copy to our storage path
        copy2(train_file.name, self.data_dir)
        copy2(test_file.name, self.data_dir)
        self.train_file_path = os.path.join(self.data_dir, train_file_name)
        self.test_file_path = os.path.join(self.data_dir, test_file_name)
        self.task_type = task_type
        self.target_column_name = target_column_name
        return "Files Submitted for processing, switch to the next tab to see the outputs"

    def build_model(self):
        fake_model = lambda x: f"{x}"
        return fake_model



class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_pred_file_path = None
        self.test_pred_file_path = None
        self.prediction_column_name = None
        self.trained_model_path = None
        self.train_df = None

        self.lit_gradio = LitGradio()
        self.evidently_data_analysis = EvidentlyDataAnalysis() # default setting to be changed later

        self.data_report_viewer = StaticPageViewer(self.evidently_data_analysis.report_parent_path)


        self.model_trainer = ModelTrainer()

        self.evidently_model_analysis = EvidentlyModelAnalysis()  # default setting to be changed later
        self.model_report_viewer = StaticPageViewer(self.evidently_model_analysis.report_parent_path)


    def run(self):
        self.lit_gradio.run()
        condition = self.lit_gradio.train_file_path != None and \
                    self.lit_gradio.test_file_path != None and \
                    self.lit_gradio.target_column_name != None and \
                    self.lit_gradio.task_type != None
        if condition:
            # assigning as per selection
            self.evidently_data_analysis.task_type = self.lit_gradio.task_type 
            self.evidently_data_analysis.train_dataframe_path=self.lit_gradio.train_file_path
            self.evidently_data_analysis.test_dataframe_path=self.lit_gradio.test_file_path
            self.evidently_data_analysis.target_column_name=self.lit_gradio.target_column_name
            self.evidently_data_analysis.run()

            # execute model trainer component here
            self.model_trainer.task_type = self.lit_gradio.task_type 
            self.model_trainer.train_dataframe_path=self.lit_gradio.train_file_path
            self.model_trainer.test_dataframe_path=self.lit_gradio.test_file_path
            self.model_trainer.target_column_name=self.lit_gradio.target_column_name
            self.model_trainer.run()

            if self.model_trainer.model_file_path != None:
                self.evidently_model_analysis.task_type = self.lit_gradio.task_type 
                self.evidently_model_analysis.train_dataframe_path=self.model_trainer.train_dataframe_path_with_preds
                self.evidently_model_analysis.test_dataframe_path=self.model_trainer.test_dataframe_path_with_preds
                self.evidently_model_analysis.target_column_name=self.lit_gradio.target_column_name
                self.evidently_model_analysis.prediction_column_name = 'prediction'
                # run the model analysis
                self.evidently_model_analysis.run()
                # serve model

    def configure_layout(self):
        tabs = [
            {"name": "Experiment Configuration", "content": self.lit_gradio},
            {"name": "Data Analysis", "content": self.data_report_viewer},
            {"name": "Model Analysis", "content": self.model_report_viewer}
        ]

        return tabs



if __name__ == "__main__":
    app = L.LightningApp(RootFlow())
