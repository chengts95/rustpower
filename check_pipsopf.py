import os, pandapower as pp
pp_path = os.path.dirname(pp.__file__)
f = pp_path+'/pypower/pipsopf_solver.py'
src = open(f).read()
idx = src.find('pips(')
print(src[max(0,idx-300):idx+600])
