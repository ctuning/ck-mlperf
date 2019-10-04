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

    image_list_filename = dep_env('images', 'CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF')
    source_dir          = dep_env('images', 'CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR')
    preprocessed_ext    = dep_env('images', 'CK_ENV_DATASET_IMAGENET_PREPROCESSED_NEW_EXTENSION')

    image_count         = int(my_env('CK_LOADGEN_DATASET_SIZE')) if has_env('CK_LOADGEN_DATASET_SIZE') else int(my_env('CK_BATCH_SIZE')) * int(my_env('CK_BATCH_COUNT'))
    images_offset       = int(my_env('CK_SKIP_IMAGES') or '0')

    sorted_filenames    = [filename for filename in sorted(os.listdir(source_dir)) if filename.lower().endswith('.' + preprocessed_ext) ]

    selected_filenames  = sorted_filenames[images_offset:images_offset+image_count] if image_count else sorted_filenames[images_offset:]

    selected_var_paths  = [ os.path.join("$<<CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR>>$", filename) for filename in selected_filenames ]

    with open(image_list_filename, 'w') as f:
        for filename in selected_filenames:
            f.write(filename + '\n')

    print('=-=-=-=-=- done.\n')

    return {
        'return': 0,
        'new_env': {},
        'run_input_files': [ '$<<>>$' + image_list_filename ] + selected_var_paths,
        'run_output_files': [],
    }
