#
# Collective Knowledge - common functionality for MLPerf benchmarking.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developers:
# - Leo Gordon, leo@dividiti.com
# - Anton Lokhmotov, anton@dividiti.com
#

cfg={}  # Will be updated by CK (meta description of this module)
work={} # Will be updated by CK (temporal data)
ck=None # Will be updated by CK (initialized CK kernel)

import getpass
import os
import sys
from pprint import pprint


def init(i):
    """
    Not to be called directly.
    """

    return {'return':0}


def compare_experiments(i):
    """
    Input:  {
                (cids[])            - up to 2 CIDs of entries to compare (interactive by default)
                (repo_uoa)          - experiment repository ('*' by default)
                (extra_tags)        - extra tags to search for CIDs
            }

    Output: {
                return              - return code =  0, if successful
                                                  >  0, if error
                (error)             - error text if return > 0
            }
    """
    cids = i.get('cids')
    repo_uoa = i.get('repo_uoa', '*')
    extra_tags = i.get('extra_tags', 'mlperf,accuracy')
    # Return an error if more than 2 CIDs have been provided.
    if len(cids) > 2:
        return {'return':1, 'error': 'only support up to 2 CIDs'}

    # Interactively select experiment entries if fewer than 2 CIDs have been provided.
    for i in range(len(cids),2):
        ck.out( 'Pick experiment #{} for comparison:'.format(i) )
        pick_exp_adict = { 'action':       'pick_an_experiment',
                           'module_uoa':   'mlperf',
                           'repo_uoa':     repo_uoa,
                           'extra_tags':   extra_tags,
        }
        r=ck.access( pick_exp_adict )
        if r['return']>0: return r
        cids.append( r['cid'] )

    # Print the two CIDs to compare.
    ck.out( '-----------------------------------' )
    ck.out( 'The pair of experiments to compare:' )
    for cid in cids:
        ck.out( '- ' + cid )

    return {'return':0}

def list_experiments(i):
    """
    Input:  {
                (repo_uoa)          - experiment repository name ('*' by default)
                (extra_tags)        - extra tags to filter
                (add_meta)          - request to return metadata with each experiment entry
            }

    Output: {
                return              - return code =  0, if successful
                                                  >  0, if error
                (error)             - error text if return > 0
            }
    """

    repo_uoa        = i.get('repo_uoa', '*')
    extra_tags      = i.get('extra_tags')
    all_tags        = 'mlperf' + ( ',' + extra_tags if extra_tags else '' )
    add_meta        = i.get('add_meta')

    search_adict    = { 'action':       'search',
                        'repo_uoa':     repo_uoa,
                        'module_uoa':   'experiment',
                        'data_uoa':     '*',
                        'tags':         all_tags,
                        'add_meta':     add_meta,
    }
    r=ck.access( search_adict )
    if r['return']>0: return r

    list_of_experiments = r['lst']

    repo_to_names_list = {}
    for entry in list_of_experiments:
        repo_uoa    = entry['repo_uoa']
        data_uoa    = entry['data_uoa']
        if not repo_uoa in repo_to_names_list:
            repo_to_names_list[ repo_uoa ] = []
        repo_to_names_list[ repo_uoa ] += [ data_uoa ]

    if i.get('out')=='con':
        for repo_uoa in repo_to_names_list:
            experiments_this_repo = repo_to_names_list[ repo_uoa ]
            ck.out( '{} ({}) :'.format(repo_uoa, len(experiments_this_repo) ) )
            for data_uoa in experiments_this_repo:
                ck.out( '\t' + data_uoa )
            ck.out( '' )

    return {'return':0, 'lst': list_of_experiments, 'repo_to_names_list': repo_to_names_list}


def pick_an_experiment(i):
    """
    Input:  {
                (repo_uoa)          - experiment repository ('*' by default)
                (extra_tags)        - extra tags to filter
            }

    Output: {
                return              - return code =  0, if successful
                                                  >  0, if error
                (error)             - error text if return > 0
            }
    """

    repo_uoa        = i.get('repo_uoa', '*')
    extra_tags      = i.get('extra_tags')

    list_exp_adict  = { 'action':       'list_experiments',
                        'module_uoa':   'mlperf',
                        'repo_uoa':     repo_uoa,
                        'extra_tags':   extra_tags,
    }
    r=ck.access( list_exp_adict )
    if r['return']>0: return r

    if len(r['lst'])==0:
        return {'return':1, 'error':'No experiments to choose from - please relax your filters'}

    all_experiment_names = [ '{repo_uoa}:{module_uoa}:{data_uoa}'.format(**entry_dict) for entry_dict in r['lst']]

    number_of_experiments = len(all_experiment_names)
    select_adict = {'action': 'select_string',
                    'module_uoa': 'misc',
                    'options': all_experiment_names,
                    'default': str(number_of_experiments-1),
                    'question': 'Please select one of the experiment entries',
    }
    r=ck.access( select_adict )
    if r['return']>0:
        return r
    else:
        cid = r['selected_value']

    return {'return':0, 'cid': cid}

