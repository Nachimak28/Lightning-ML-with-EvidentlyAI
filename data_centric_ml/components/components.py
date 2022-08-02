import os
import tempfile
import lightning as L
from lightning.app.frontend.web import StaticWebFrontend
from lightning.app.storage.path import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from .utils import construct_pred_file_names

class StaticPageViewer(L.LightningFlow):
    def __init__(self, page_path: str):
        super().__init__()
        self.serve_dir = page_path

    def configure_layout(self):
        return StaticWebFrontend(serve_dir=self.serve_dir)


class ModelTrainer(L.LightningWork):
    def __init__(self, train_dataframe_path=None, 
                test_dataframe_path=None, 
                target_column_name=None, 
                prediction_column_name='prediction', 
                task_type='classification', 
                parallel=True) -> None:
        
        super().__init__(parallel=parallel)

        self.data_dir = Path(tempfile.mkdtemp())
        self.train_dataframe_path = train_dataframe_path
        self.test_dataframe_path = test_dataframe_path
        self.target_column_name = target_column_name
        self.prediction_column_name = prediction_column_name
        self.task_type = task_type


        self.model_file_path = None
        self.train_dataframe_path_with_preds = None
        self.test_dataframe_path_with_preds = None

    def run(self):
        if self.task_type == 'classification':
            model = RandomForestClassifier(random_state=42)
        elif self.task_type == 'regression':
            model = RandomForestRegressor(random_state=42)
        else:
            raise Exception('Invalid Task Type')
        
        # read data
        train_df = pd.read_csv(self.train_dataframe_path)
        test_df = pd.read_csv(self.test_dataframe_path)
        # prepare data for training
        feature_columns = list(train_df)
        feature_columns.remove(self.target_column_name)


        model.fit(train_df[feature_columns], train_df[self.target_column_name])

        # saving trained model just in case
        model_filename = os.path.join(self.data_dir, f'Finalized_model_{self.task_type}.sav')
        joblib.dump(model, model_filename)
        

        # add predictions to the dataframe
        train_df['prediction']  = model.predict(train_df[feature_columns])
        test_df['prediction'] = model.predict(test_df[feature_columns])

        # process filenames to save
        train_df_parent_dir, train_pred_file_path = construct_pred_file_names(self.train_dataframe_path)
        test_df_parent_dir, test_pred_file_path = construct_pred_file_names(self.test_dataframe_path)

        self.train_dataframe_path_with_preds = os.path.join(self.data_dir, train_pred_file_path)
        self.test_dataframe_path_with_preds = os.path.join(self.data_dir, test_pred_file_path)
        
        # save files with preds
        train_df.to_csv(self.train_dataframe_path_with_preds, index=False)
        test_df.to_csv(self.test_dataframe_path_with_preds, index=False)

        # assign model at last to signal other component that its ready for serving
        self.model_file_path = model_filename