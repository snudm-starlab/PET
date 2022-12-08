################################################################################
# Starlab Transformer Compression with PET (Parameter-Efficient Knowledge Distillation on Transformer)
#
# Author: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Nov 29, 2022
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/facebookresearch/fairseq
################################################################################
import importlib
import os

# recursively and automatically import any Python files in the current directory
cur_dir = os.path.dirname(__file__)
if cur_dir == '':
    cur_dir = '.'
tar_dirs = ['criterions']


def import_py_files(base_dir, _dir):
    dir_path = os.path.join(base_dir, _dir)

    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)

        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py"))
        ):
            mod_name = file[: file.find(".py")] if file.endswith(".py") else file
            path_name = _dir.replace('/', '.')
            module = importlib.import_module(f'{__name__}.{path_name}.{mod_name}')


for tar_dir in tar_dirs:
    import_py_files(cur_dir, tar_dir)