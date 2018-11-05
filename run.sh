#!/bin/bash
export seed=0       # seed to start pseudo number generation
export timelimit=1  # total execution time for the program in hours
export logafter=100 # number of iterations of algorithm 1c (1-city-at-a-time distortion/VOCAL paper) between logs
export budget=100   # total time available for the trip (includes probing time)
export mode="off"   # off (static) or on (dynamic) - line version

export avoidnumpywarnings=`grep -v gaussian_process | grep -v np.newaxis | grep -v convergence_dict | grep -v warnings.warn`

for function in `seq 1 10`
  do for gridsize in `echo 10 7 4`
    do for algorithm in `echo 1c sw`    # sw = Particle Swarm Optmization (PSO parameters should be choosen inside swarm.py; currently 4000 iterations and pop. size 100)
      do time python -u ocean.py dontplot $seed $timelimit $logafter $gridsize $budget $function $algorithm $mode 2> >(avoidnumpywarnings) | grep res: | tee f$function-$seed-${gridsize}x$gridsize-$budget-$algorithm-$timelimit.$mode.log
    done
  done
done

