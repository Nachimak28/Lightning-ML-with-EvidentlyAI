from hashlib import new
import os
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


parser = argparse.ArgumentParser(description='Train a simple model for serving')
parser.add_argument('--train_dataframe_path', type=str,
                    help='Train dataframe path')

parser.add_argument('--test_dataframe_path', type=str,
                    help='Test dataframe path')

parser.add_argument('--target_column_name', type=str,
                    help='Target Column Name')

parser.add_argument('--task_type', type=str,
                    help='Task Type')

args = parser.parse_args()

train_df_path = args.train_dataframe_path
test_df_path = args.test_dataframe_path
target_column_name = args.target_column_name
task_type = args.task_type

print(train_df_path)
print(test_df_path)
print(task_type)
# preprocessing step here as per your need - we skip here for demo purposes

if task_type == 'classification':
    model = RandomForestClassifier(random_state=42)
elif task_type == 'regression':
    model = RandomForestRegressor(random_state=42)
else:
    raise Exception('Invalid Task Type')


print(train_df_path)
print(test_df_path)
print(task_type)

# read data
train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)
# prepare data for training
feature_columns = list(train_df)
feature_columns.remove(target_column_name)


model.fit(train_df[feature_columns], train_df[target_column_name])

# saving trained model just in case
filename = f'Finalized_model_{task_type}.sav'
joblib.dump(model, filename)
print('Model ready to serve')

# add predictions to the dataframe
train_df['prediction']  = model.predict(train_df[feature_columns])
test_df['prediction'] = model.predict(test_df[feature_columns])

# train parent path
train_df_parent_dir, train_df_file_name = os.path.split(train_df_path)

# test parent path
test_df_parent_dir, test_df_file_name = os.path.split(test_df_path)


def construct_pred_file_names(file_path):
    residing_dir, file_name_with_ext = os.path.split(file_path)
    file_name, ext = os.path.splitext(file_name_with_ext)
    new_file_name_with_preds = file_name + '_preds'
    return residing_dir, new_file_name_with_preds + ext

train_df_parent_dir, train_pred_file_path = construct_pred_file_names(train_df_path)
test_df_parent_dir, test_pred_file_path = construct_pred_file_names(test_df_path)


# write dataframes to designated path
train_df.to_csv(os.path.join(train_df_parent_dir, train_pred_file_path))

test_df.to_csv(os.path.join(test_df_parent_dir, test_pred_file_path))


