import matrix_io
from sklearn import preprocessing
import numpy as np
import pandas as pd


class DataSets:
    root_dir = ".."
    data_sets = {
        'colon': (
            {
                "path": "/COLON/COLON/colon.data"
            },
            {
                "path": "/COLON/COLON/colon.labels",
                "apply_transform": np.sign
            }
        ),
        'arcene': (
            {
                "path": "/ARCENE/ARCENE/arcene.data",
                "apply_transform": np.transpose
            },
            {
                'path': "/ARCENE/ARCENE/arcene.labels",
            }
        ),
        'dexter': (
            {
                "path": "/DEXTER/DEXTER/dexter.data",
                "method": "sparse_matrix",
                "args": [20000]
            },
            {
                "path": "/DEXTER/DEXTER/dexter.labels",
            }
        ),
        "dorothea": (
            {
                "path": "/DOROTHEA/DOROTHEA/dorothea.data",
                "method": "sparse_binary_matrix",
                "args": [100001]
            },
            {
                "path": "/DOROTHEA/DOROTHEA/dorothea.labels",
            }
        )
    }

    @staticmethod
    def load(name):
        data_directory, labels_directory = DataSets.data_sets[name]
        data = DataSets.__load_data_set_file(data_directory)
        labels = DataSets.__load_data_set_file(labels_directory)
        data_scaled = preprocessing.scale(data)
        return data_scaled, labels

    @staticmethod
    def __load_data_set_file(info):
        data = getattr(matrix_io, info.get('method', 'regular_matrix'))(
            DataSets.root_dir + info['path'],
            *info.get('args', []),
            **info.get('kwargs', {})
        )
        apply_transform = info.get('apply_transform', False)
        if apply_transform:
            return apply_transform(data)
        return data


class Weights:
    def load(data_set, cv, assessment_method, feature_method):
        try:
            filename = Weights.file_name(data_set, cv, assessment_method, feature_method) + ".npy"
            weights = np.load(filename)
            return weights
        except FileNotFoundError:
            print("File " + filename + " not found")
            raise

    @staticmethod
    def file_name(data_set, cv, assessment_method, feature_method):
        return Weights.dir_name(data_set, cv, assessment_method) + "/" + feature_method.__name__

    @staticmethod
    def dir_name(data_set, cv, method):
        return "{root_dir}/feature_{method}s/{data_set}/{cv}".format(
            root_dir=DataSets.root_dir,
            method=method,
            data_set=data_set,
            cv=type(cv).__name__
        )


class Analysis:
    def load_csv(data_set, cv, assessment_method, feature_method):
        try:
            filename = Analysis.file_name(data_set, cv, assessment_method, feature_method) + ".csv"
            stats = pd.read_csv(filename)
            return stats
        except FileNotFoundError:
            print("File " + filename + " not found")
            raise

    @staticmethod
    def file_name(data_set, cv, assessment_method, feature_method):
        return Analysis.dir_name(data_set, cv, assessment_method) + "/" + feature_method.__name__

    @staticmethod
    def dir_name(data_set, cv, method):
        return "{root_dir}/{method}s_analyse/{data_set}/{cv}".format(
            root_dir=DataSets.root_dir,
            method=method,
            data_set=data_set,
            cv=type(cv).__name__
        )
