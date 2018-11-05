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
    run.sh  # Parameters are documented inside the script.
```
The program will output the total variance value, the total error among other values as a function of time.
The kernel is selected by k-fold CV.
The plot of the path or the surface of functions can be requested at the command line.
