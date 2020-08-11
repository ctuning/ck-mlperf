#
# Collective Knowledge (checking and installing software)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer: Leo Gordon, leo@dividiti.com
#

cfg={}  # Will be updated by CK (meta description of this module)
work={} # Will be updated by CK (temporal data)
ck=None # Will be updated by CK (initialized CK kernel)


def init(i):
    """
    Not to be called directly. Sets the path to the vqe_plugin.
    """

    return {'return':0}


def run(i):
    """
    Input:  {
                data_uoa            - specific entry that contains recipe for building the command
            }

    Output: {
                return              - return code =  0, if successful
                                                  >  0, if error
                (error)             - error text if return > 0
            }
    Test:
            ck run buildcmd:ls
    """

    import re
    import os
    from pprint import pprint

    interactive     = i.get('out')=='con'

    params = i.copy()
    for unwanted_key in ('action', 'repo_uoa', 'module_uoa', 'data_uoa', 'cid', 'cids', 'xcids', 'out'):
        params.pop(unwanted_key, None)

    load_adict = {  'action':           'load',
                    'repo_uoa':         i.get('repo_uoa', ''),
                    'module_uoa':       i['module_uoa'],
                    'data_uoa':         i['data_uoa'],
    }
    r=ck.access( load_adict )
    if r['return']>0: return r

    entry_dict = r['dict']
    if interactive:
        print("Entry contents:")
        pprint(entry_dict)
        print("")

    build_map   = entry_dict['build_map']
    accu        = {}

    # Accumulating values:
    #
    for param_name in params:
        param_value = params[param_name]
        accu_map = build_map[param_name][param_value]
        for accu_name in accu_map:
            if accu_name not in accu:
                accu[accu_name] = []    # manual vivification
            accu[accu_name].append( accu_map[accu_name] )
    if interactive:
        print("Accu contents:")
        pprint(accu)
        print("")

    # Substitute the accumulated values into command template:
    #
    cmd_template    = entry_dict['cmd_template']
    cmd             = cmd_template

    for match in re.finditer('(<<<(\w+)(.?)>>>)', cmd_template):
        expression, accu_name, accu_sep = match.group(1), match.group(2), match.group(3)
        if accu_name in accu:
            cmd = cmd.replace(expression, accu_sep.join(accu[accu_name]) )
        else:
            print("WARNING: Could not map {} from the dictionary, skipping it".format(expression) )
            cmd = cmd.replace(expression, '')

    if interactive:
        print("Executing the command:\n\t{}".format(cmd))
        print("")
    os.system(cmd)

    return { 'return': 0, 'cmd': cmd }


def show(i):
    """
    Input:  {
                (data_uoa)          - name of the SUT entry
            }

    Output: {
                return              - return code =  0, if successful
                                                  >  0, if error
                (error)             - error text if return > 0
            }
    Test:
            ck show sut:velociti
    """

    from pprint import pprint

    interactive     = i.get('out')=='con'
    data_uoa = i.get('data_uoa')
    if data_uoa:
        load_adict = {  'action':           'load',
                        'module_uoa':       i['module_uoa'],
                        'data_uoa':         data_uoa,
        }
        r=ck.access( load_adict )
        if r['return']>0: return r

        entry_dict = r['dict']

        if interactive:
            pprint(entry_dict)

    return { 'return': 0 }

