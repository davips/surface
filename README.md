# surface
Optimization and machine learning for sea bed exploration

# install
```
git clone https://github.com/oceanbed/surface
cd surface
```
Copy gurobi folder (gurobi752 ?) to the current folder, if exact search is needed.

# usage example (linux bash)
```
seed=0          # seed to start pseudo number generation
timelimit=1     # total execution time for the program
logafter=100    # number of iterations of algorithm 1c (i-city distortion/VOCAL paper) between logs
budget=100      # total time available for the trip (includes probing time)
mode="off"      # off (static) or on (dynamic) - line version
alias avoidnumpywarnings='e -v gaussian_process | e -v np.newaxis | e -v convergence_dict | e -v warnings.warn'
for function in `seq 1 10`; do for gridsize in `echo 10 7 4`; do for algorithm in `echo 1c sw`
    do time python -u ocean.py dontplot $seed $timelimit $logafter $gridsize $budget $function $algorithm $mode 2> >(avoidnumpywarnings) | e res: | tee f$function-$seed-${gridsize}x$gridsize-$budget-$algorithm-$timelimit.$mode.log
done; done; done
```
The program will output the total variance value, the total error among other values as a function of time.
The kernel is selected by k-fold CV.
The plot of the path or the surface of functions can be requested at the command line.
