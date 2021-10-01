import os
import time
import shutil
from zipfile import ZipFile

"""
Helper functions not associated with a particular
class are defined here for modularity.
Referenced from https://github.com/alwaysai/image-capture-dashboard
"""
STATIC = 'static'

def folder_set_up(folder):
    """
    Creates a new folder from the specified parameter 'folder'
    if it does not already exist.
    """
    if not os.path.exists(STATIC):
        os.mkdir(STATIC)

    if not os.path.exists(folder):
        os.mkdir(folder)

def get_all_files(folder):
    """
    Adds all files in the folder to a list.
    """
    all_files = []
    if not os.path.exists(folder):
        return None
    else:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.isfile(os.path.join(root, f)) and '.DS_Store' not in f:
                    all_files.append(f)
        all_files = sorted(all_files)
        return all_files


def get_file(filename, folder):
    if not os.path.exists(folder):
        return None
    else:
        for root, _, files in os.walk(folder):
            for file_set_up in files:
                if file_set_up == filename:
                    return os.path.join(root, file_set_up)

    return None

def open_dataset(folder, dataset):
    """
    Unzips zipped dataset unless the folder already exists.
    """
    # Check if folder is not empty (zipped files have already been extracted)
    # TODO: edit statement to check if correct folders exist (Annotations, JPEGImages)
    if len(os.listdir(folder)) > 0:
        print('Files have already been extracted from ' + dataset)
        return

    # opening the zip file in READ mode
    with ZipFile(dataset, 'r') as zip:
        # extracting all the files
        print('Extracting files from ' + dataset)
        zip.extractall(folder)
