import matrix_io


class DataSets:
    data_sets = {
        'arcene': (
            {
                "path": "/ARCENE/ARCENE/arcene_train.data",
            },
            {
                'path': "/ARCENE/ARCENE/arcene_train.labels",
            }
        ),
        'dexter': (
            {
                "path": "/DEXTER/DEXTER/dexter_train.data",
                "method": "sparse_matrix",
                "args": [20000]
            },
            {
                "path": "/DEXTER/DEXTER/dexter_train.labels",
            }
        ),
        "dorothea": (
            {
                "path": "/DOROTHEA/DOROTHEA/dorothea_train.data",
                "method": "sparse_binary_matrix",
                "args": [100001]
            },
            {
                "path": "/DOROTHEA/DOROTHEA/dorothea_train.labels",
            }
        )
    }

    def __init__(self, data_dir='..'):
        self.data_folder = data_dir

    def load(self, name):
        data, labels = self.data_sets[name]
        return self.__load_data_set_file(data), self.__load_data_set_file(labels)

    def __load_data_set_file(self, info):
        return getattr(matrix_io, info.get('method', 'regular_matrix'))(
            self.data_folder + info['path'],
            *info.get('args', []),
            **info.get('kwargs', {})
        )
