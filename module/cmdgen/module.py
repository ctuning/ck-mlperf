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


def gen(i):
    """
    Input:  {
                data_uoa            - specific entry that contains recipe for building the command
                (any_map_params)    - any command line parameters (supported by build_map)
            }

    Output: {
                return              - return code =  0, if successful
                                                  >  0, if error
                (error)             - error text if return > 0
            }
    Test:
            ck gen cmdgen:ls
            ck gen cmdgen:ls --mode=long --home
    """

    import re
    from pprint import pprint

    interactive     = i.get('out')=='con'

    input_params = i.copy()
    for unwanted_key in ('action', 'repo_uoa', 'module_uoa', 'data_uoa', 'cid', 'cids', 'xcids', 'out'):
        input_params.pop(unwanted_key, None)

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
    for param_name in input_params:
        param_value = input_params[param_name]

        if param_name in build_map:
            # Start with the specific value, but fallback to default:
            accu_map = build_map[param_name].get(param_value) or build_map[param_name]['###']

            for accu_name in accu_map:
                if accu_name not in accu:
                    accu[accu_name] = []    # manual vivification

                accu_value_list = accu_map[accu_name]
                if type(accu_value_list)!=list:
                    accu_value_list = [ accu_value_list ]

                for accu_value in accu_value_list:
                    accu[accu_name].append( accu_value.replace('###', param_value) )

    if interactive:
        print("Accu contents:")
        pprint(accu)
        print("")

    # Substitute the accumulated values into command template:
    #
    cmd_template    = entry_dict['cmd_template']
    cmd             = cmd_template

    for match in re.finditer('(<<<(\??)(\w+)(.?)>>>)', cmd_template):
        expression, optional, anchor_name, accu_sep = match.group(1), match.group(2), match.group(3), match.group(4)
        if anchor_name in accu:
            if anchor_name in input_params:
                return {'return':1, 'error':"Both input_params and accu contain '{}' anchor, ambiguous substitution".format(anchor_name)}
            else:
                cmd = cmd.replace(expression, accu_sep.join(accu[anchor_name]) )
        elif anchor_name in input_params:
            cmd = cmd.replace(expression, accu_sep.join(input_params[anchor_name]) )
        elif optional=='?':
            cmd = cmd.replace(expression, '')
        else:
            return {'return':1, 'error':"Neither input_params nor accu contain substitution for non-optional '{}' anchor".format(anchor_name)}

    if interactive:
        print("The generated command:\n\t{}".format(cmd))

    return { 'return': 0, 'cmd': cmd }


def run(i):
    """
    Input:  {
                data_uoa            - specific entry that contains recipe for building the command
                (any_map_params)    - any command line parameters (supported by build_map)
            }

    Output: {
                return              - return code =  0, if successful
                                                  >  0, if error
                (error)             - error text if return > 0
            }
    Test:
            ck run cmdgen:ls
            ck run cmdgen:ls --mode=long --home
    """

    import os

    i['action'] = 'gen'
    r=ck.access( i )
    if r['return']>0: return r

    cmd = r['cmd']

    os.system(cmd)

    return { 'return': 0 }


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

