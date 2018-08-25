# surface
Optimization and machine learning for sea bed exploration

# install
```
git clone https://github.com/oceanbed/surface
cd surface
```
Copy gurobi folder (gurobi752 ?) to the current folder, if exact search is needed.

# usage examples
```
python -u ocean.py
python -u ocean.py known=4x4 next=maxvar distortion=random search=heuri log=mini verbosity=less plot=fun f=2
python -u ocean.py known=4x4 next=maxvar distortion=random search=heuri log=mini verbosity=less plot=path f=2
python -u ocean.py known=4x4 next=maxvar distortion=pswarm search=heuri log=full verbosity=less plot=none f=2 2> /dev/null | grep out
```
The program will output the total variance value, the total error, the number of probings and the name of the kernel selected by k-fold CV. The plot of the path or the surface of functions can be requested at the command line.
