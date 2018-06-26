from trip import *


def parse_args(argv):
    try:
        opts = dict(tuplefy(argv[1:]))
        at_random = opts['first-points'] == 'rnd'
        full_log = opts['log'] == 'full'
        swarm = opts['distortion'] == 'swarm'
        if opts['distortion'] == 'rnd':
            distortionf = random_distortion
        elif opts['distortion'] == 'shrink':
            distortionf = median_distortion
        else:
            distortionf = no_distortion
        search_type = opts['search'] == 'exact'
    except:
        print('============================================================')
        print('Usage:\npython ocean.py option1 value1 option2 value2 ... optionN valueN\nOptions:'.expandtabs(20))
        print('============================================================')
        print('Option\tvalues\tdescription'.expandtabs(20))
        print('------------------------------------------------------------')
        print('log\tfull|mini\tcontrols amount of log output: maximal or just the variance'.expandtabs(20))
        print('first-points\trnd|maxvar\tselect inicial probing points at random or according to max variance'.expandtabs(20))
        print('distortion\tswarm|rnd|shrink\ttype of distortion to apply to the points'.expandtabs(20))
        print('search\texact|heur\ttype of search'.expandtabs(20))
        print('============================================================')
        sleep(1)
        raise
    else:
        print(opts)
    return at_random, full_log, swarm, distortionf, search_type
