#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer(s):
# - Anton Lokhmotov, anton@dividiti.com
#

import os

##############################################################################
# setup environment setup

def setup(i):
    """
    Input:  {
              cfg              - meta of this soft entry
              self_cfg         - meta of module soft
              ck_kernel        - import CK kernel module (to reuse functions)

              host_os_uoa      - host OS UOA
              host_os_uid      - host OS UID
              host_os_dict     - host OS meta

              target_os_uoa    - target OS UOA
              target_os_uid    - target OS UID
              target_os_dict   - target OS meta

              target_device_id - target device ID (if via ADB)

              tags             - list of tags used to search this entry

              env              - updated environment vars from meta
              customize        - updated customize vars from meta

              deps             - resolved dependencies for this soft

              interactive      - if 'yes', can ask questions, otherwise quiet
            }

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0

              bat          - prepared string for bat file
            }

    """

    import os

    # Get variables
    ck=i['ck_kernel']
    s=''

    iv=i.get('interactive','')

    cus=i.get('customize',{})
    full_path=cus.get('full_path','')

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']

    winh=hosd.get('windows_base','')

    env=i['env']
    ep=cus['env_prefix']

    loadgen_dir     = os.path.dirname( full_path )
    inference_dir   = os.path.dirname( loadgen_dir )

    env[ep] = inference_dir

    for ver_subdir in ('v0.5', 'v0.7', 'vision'):
        ver_suffix  = ver_subdir.replace('.','').upper()
        ver_fulldir = os.path.join(inference_dir, ver_subdir)
        mlperf_candidate    = os.path.join( ver_fulldir, 'mlperf.conf' )
        if os.path.exists( mlperf_candidate ):
            env[ep+'_MLPERF_CONF'] = mlperf_candidate

        if os.path.isdir( ver_fulldir ):
            env[ep + '_' + ver_suffix] = env[ep + '_' + 'VLATEST'] = ver_fulldir
            python_dir = os.path.join( ver_fulldir, 'classification_and_detection', 'python' )

    mlperf_candidate    = os.path.join( inference_dir, 'mlperf.conf' )
    if os.path.exists( mlperf_candidate ):
        env[ep+'_MLPERF_CONF'] = mlperf_candidate

    env[ep+'_LOADGEN']  = loadgen_dir
    env['PYTHONPATH']   = python_dir + ( ';%PYTHONPATH%' if winh=='yes' else ':${PYTHONPATH}')

    return {'return':0, 'bat':s}
