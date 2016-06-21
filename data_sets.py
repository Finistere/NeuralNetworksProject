import matrix_io
from sklearn import preprocessing
import numpy as np


class DataSets:
    root_dir = ".."
    data_sets = {
        'colon': (
            {
                "path": "/Colon/data.txt"
            },
            {
                "path": "/Colon/labels.txt",
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
        return data, labels

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
