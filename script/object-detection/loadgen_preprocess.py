#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
from shutil import copy2


def set_up_intel_cpu_for_server_or_singlestream_scenario(i):

    env         = i['env']
    scenario    = env['CK_LOADGEN_SCENARIO']

    if scenario.lower() in ('server', 'singlestream'):
        print('\n-=-=-=-=-= Setting up the Intel CPU for the test (scenario = {})'.format(scenario))

        command_list = [
            'rm -rf /dev/shem/*',
            'echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct',
            'sync',
            'echo 1 | sudo tee /proc/sys/vm/compact_memory',
            'echo 3 | sudo tee /proc/sys/vm/drop_caches',
        ]

        for cmd in command_list:
            os.system( cmd )

    else:
        print('\n-=-=-=-=-= NOT setting up the Intel CPU for the test (scenario = {})'.format(scenario))

    print('=-=-=-=-=- done.\n')

    return {'return':0}


def user_conf_and_audit_config(i):

    def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
    deps=i['deps']
    env=i['env']

    # Try ML_MODEL_MODEL_NAME from the OpenVINO model first. If unavailable, try MODEL_NAME from its 'model-source' dependency.
    model_name = dep_env('weights', 'ML_MODEL_MODEL_NAME') or deps['weights']['dict']['deps']['model-source']['dict']['customize']['install_env']['MODEL_NAME']
    print('\n-=-=-=-=-= Generating user.conf for model "{}" ...'.format(model_name))
    scenario            = env['CK_LOADGEN_SCENARIO']
    user_conf_rel_path  = env['CK_LOADGEN_USER_CONF']
    user_conf           = []
    env_to_conf         = {
        'CK_LOADGEN_MAX_QUERY_COUNT':                   ('max_query_count', 1),
        'CK_LOADGEN_BUFFER_SIZE':                       ('performance_sample_count_override', 1),
        'CK_LOADGEN_SAMPLES_PER_QUERY':                 ('samples_per_query', 1),
        'CK_LOADGEN_TARGET_LATENCY':                    ('target_latency', 1),
        'CK_LOADGEN_TARGET_QPS':                        ('target_qps', 1),

        'CK_LOADGEN_MAX_DURATION_S':                    ('max_duration_ms', 1000),
        'CK_LOADGEN_OFFLINE_EXPECTED_QPS':              ('offline_expected_qps', 1),
    }
    for env_key in env_to_conf.keys():
        if env_key in env:
            orig_value = env[env_key]
            (config_category_name, multiplier) = env_to_conf[env_key]
            new_value = orig_value if multiplier==1 else float(orig_value)*multiplier
            
            user_conf.append("{}.{}.{} = {}\n".format(model_name, scenario, config_category_name, new_value))

    # Write 'user.conf' into the current directory ('tmp').
    user_conf_abs_path = os.path.join(os.path.abspath(os.path.curdir), user_conf_rel_path)
    with open(user_conf_abs_path, 'w') as user_conf_file:
         user_conf_file.writelines(user_conf)

    # Copy 'audit.config' for compliance testing into the current directory ('tmp').
    compliance_test_config = 'audit.config'
    compliance_test = env.get('CK_MLPERF_COMPLIANCE_TEST','')
    inference_root = dep_env('mlperf-inference-src', 'CK_ENV_MLPERF_INFERENCE')
    if compliance_test != '' and inference_root != '':
        if compliance_test in [ 'TEST01', 'TEST04-A', 'TEST04-B', 'TEST05' ]:
            compliance_test_source_dir = os.path.join(inference_root, 'compliance', 'nvidia', compliance_test)
            if compliance_test in [ 'TEST01' ]: compliance_test_source_dir = os.path.join(compliance_test_source_dir, model_name)
            compliance_test_config_source_path = os.path.join(compliance_test_source_dir, compliance_test_config)
            compliance_test_config_target_path = os.path.join(os.path.abspath(os.path.curdir), compliance_test_config)
            copy2(compliance_test_config_source_path, compliance_test_config_target_path)
        else:
            raise Exception("Warning: Unsupported compliance test: '{}'".format(compliance_test))

    print('=-=-=-=-=- done.\n')

    return {'return':0}


def schindler(i):
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



# This preprocessing subroutine is a combination of several sequential processes,
# but the only non-trivial output is the output of schindler(), so we call it last:
#
def ck_preprocess(i):
    env=i['env']

    ret_dict = {'return':0}

    if env.get('CK_MLPERF_PRE_SET_UP_INTEL_CPU','').lower() in ('yes', 'on', 'true', '1'):
        ret_dict = set_up_intel_cpu_for_server_or_singlestream_scenario(i)

    if env.get('CK_MLPERF_PRE_USER_CONF_AND_AUDIT_CONFIG','').lower() in ('yes', 'on', 'true', '1'):
        ret_dict = user_conf_and_audit_config(i)

    if env.get('CK_MLPERF_PRE_SCHINDLER','').lower() in ('yes', 'on', 'true', '1'):
        ret_dict = schindler(i)

    return ret_dict

# Do not add anything here!
