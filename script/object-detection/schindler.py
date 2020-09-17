#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os

def ck_preprocess(i):
    print('\n-=-=-=-=-= Generating a list of files to be processed...')
    def has_env(var): return var in i['env']
    def my_env(var): return i['env'].get(var)
    def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
    def has_dep_env(dep, var): return var in i['deps'][dep]['dict']['env']

    index_filename  = dep_env('dataset', 'CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF')
    source_dir      = dep_env('dataset', 'CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR')
    image_count     = int(my_env('CK_LOADGEN_DATASET_SIZE')) if has_env('CK_LOADGEN_DATASET_SIZE') else int(my_env('CK_BATCH_SIZE')) * int(my_env('CK_BATCH_COUNT'))
    images_offset   = int(my_env('CK_SKIP_IMAGES') or '0')

    all_index_lines = []
    with open( os.path.join(source_dir, index_filename), "r") as i_file:
        all_index_lines = i_file.read().splitlines()

    selected_index_lines    = all_index_lines[images_offset:images_offset+image_count]

    selected_filenames      = [ line.split(';')[0] for line in selected_index_lines]

    selected_var_paths      = [ os.path.join("$<<CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR>>$", filename) for filename in selected_filenames ]

    with open(index_filename, 'w') as o_file:
        for filename in selected_index_lines:
            o_file.write(filename + '\n')

    print('=-=-=-=-=- done.\n')

    return {
        'return': 0,
        'new_env': {},
        'run_input_files': [ '$<<>>$' + index_filename ] + selected_var_paths,
        'run_output_files': [],
    }
