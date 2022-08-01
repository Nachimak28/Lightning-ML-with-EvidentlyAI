import os


def construct_pred_file_names(file_path):
    residing_dir, file_name_with_ext = os.path.split(file_path)
    file_name, ext = os.path.splitext(file_name_with_ext)
    new_file_name_with_preds = file_name + '_preds'
    return residing_dir, new_file_name_with_preds + ext