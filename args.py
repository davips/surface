#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from trip import *
from time import *
from functions import *


def parse_args(argv):
    opts = {'known': ({'4x4': 4, '7x7': 7, '10x10': 10}, 'amount of inicial probing points'),
            'next': ({'chance': True, 'maxvar': False}, 'select next probing points by chance or according to max variance'),
            'distortion': ({'random': random_distortion, 'none': no_distortion, 'pswarm': ps_distortion, 'shrink': median_distortion}, 'type of distortion to apply to the points'),
            'search': ({'exact': True, 'heuri': False}, 'type of TSP search: exaustive or heuristic'),
            'log': ({'mini': False, 'full': True}, 'controls amount of output: just the variance values or more values'),
            'verbosity': ({'less': False, 'more': True}, 'controls logging from Trip: whether to show internals information or not'),
            'plot': ({'path': 'path', 'var': 'var', 'est' :'est', 'fun': 'fun', 'none':'none'}, 'whether to plot path, variance, estimated/true function or nothing'),
            'f': ({'1': f1, '2': f2, '3': f3, '4': f4, '5': f5, '6': f6, '7': f7, '8': f8, '9': f9, '10': f10}, 'true function')}
    try:
        args = list(map(lambda x: tuple(x.split('=')), argv[1:]))
        dic = dict(args)

        # Check arguments.
        for arg, val in args:
            if val not in opts[arg][0].keys():
                raise Exception('Value ' + val + ' not in [' + '|'.join(opts[arg][0].keys()) + '].')

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
    return r['plot'], r['f'], r['known'], r['next'], r['log'], r['distortion'] == ps_distortion, r['distortion'], r['search'], r['verbosity']
