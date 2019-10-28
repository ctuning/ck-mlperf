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

    mlperf_source_repo_env = i['cfg']['deps']['mlperf-inference-src']['dict']['env']

#    from pprint import pprint
#    pprint(mlperf_source_repo_env)

    search_in_path = mlperf_source_repo_env['CK_ENV_MLPERF_INFERENCE_V05']

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
