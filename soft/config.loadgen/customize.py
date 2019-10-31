#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer: Leo Gordon, dividiti
#

import os


def dirs(i):

    config_source = i.get('install_env', {}).get('LOADGEN_CONFIG_SOURCE', 'MLPERF_INFERENCE_GIT_CHECKOUT')

    if config_source == 'MLPERF_INFERENCE_GIT_CHECKOUT':
        mlperf_source_repo_env  = i['cfg']['deps']['mlperf-inference-src']['dict']['env']
        search_in_path          = mlperf_source_repo_env['CK_ENV_MLPERF_INFERENCE_V05']
    elif config_source == 'SOFT_ENTRY_INTERNAL':
        search_in_path          = i['soft_entry_path']

#    from pprint import pprint
#    pprint(mlperf_source_repo_env)

    return {'return': 0, 'dirs': [ search_in_path ] }


def setup(i):

    cus         = i['customize']
    env         = i['env']
    full_path   = cus.get('full_path','')
    env_prefix  = cus.get('env_prefix','')

    if env_prefix!='' and full_path!='':
        audit_dir   = os.path.dirname(full_path)

        env[env_prefix]         = audit_dir
        env[env_prefix+'_FILE'] = full_path

    return {'return':0, 'bat':''}
