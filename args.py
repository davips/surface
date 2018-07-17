from trip import *
from time import *


def parse_args(argv):
    opts = {'distortion': ({'random': random_distortion, 'pswarm': no_distortion, 'shrink': median_distortion}, 'type of distortion to apply to the points'),
            'first-points': ({'chance': True, 'maxvar': False}, 'select inicial probing points by chance or according to max variance'),
            'log': ({'mini': False, 'full': True}, 'controls amount of log output: just the variance values or maximal'),
            'search': ({'exact': True, 'heuri': False}, 'type of TSP search: exaustive or heuristic')}
    try:
        args = list(map(lambda x: tuple(x.split('=')), argv[1:]))
        dic = dict(args)

        # Check arguments.
        for arg, val in args:
            if val not in opts[arg][0].keys():
                raise Exception('Value ' + val + ' not in [' + '|'.join(opts[arg][0].keys()), '].')

        r = {}
        for k in opts.keys():
            r[k] = opts[k][0][dic[k]]
    except:
        print()
        print('============================================================')
        print('Usage:\npython ocean.py option1=value1 option2=value2 ... optionN=valueN\nOptions:'.expandtabs(20))
        print('============================================================')
        print('Option\tvalues\tdescription'.expandtabs(22))
        print('------------------------------------------------------------')
        for k, (dic, d) in opts.items():
            print((k + '\t' + '|'.join(dic.keys()) + '\t' + d).expandtabs(22))
        print('============================================================')
        print()
        print()
        sleep(1)
        raise
    else:
        print(dic)
    return r['first-points'], r['log'], r['distortion'] == no_distortion, r['distortion'], r['search']
