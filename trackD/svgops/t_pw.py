import sys,time
from t_o7 import *
import core
core.render=core.render_pw
import t_o7; t_o7.render=core.render_pw
T=int(sys.argv[1]); r=run_att(T,tag=f"pw{T}"); print(r)
