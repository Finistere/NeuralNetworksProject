import matrix_io


class DataSets:
    root_dir = ".."
    data_sets = {
        'arcene': (
            {
                "path": "/ARCENE/ARCENE/arcene.data",
                "transpose": True
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
        data, labels = DataSets.data_sets[name]
        return DataSets.__load_data_set_file(data), DataSets.__load_data_set_file(labels)

    @staticmethod
    def __load_data_set_file(info):
        data = getattr(matrix_io, info.get('method', 'regular_matrix'))(
            DataSets.root_dir + info['path'],
            *info.get('args', []),
            **info.get('kwargs', {})
        )
        if info.get('transpose', False):
            return data.T
        return data
