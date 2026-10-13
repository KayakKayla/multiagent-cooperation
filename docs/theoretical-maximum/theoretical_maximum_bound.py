"""Optimistic action-flow upper bound; requires OR-Tools. No training.

This relaxes collisions, cooking delay, ordering and counter capacity.
Every legal schedule maps to a feasible flow, so its optimum is an upper bound.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import theoretical_maximum as rules
from ortools.linear_solver import pywraplp
s=pywraplp.Solver.CreateSolver('GLOP')
nodes=[(p,d,i) for p in range(8) for d in range(4) for i in range(4)]
flow={n:s.Constraint(0,0) for n in nodes}
resources={(c,i):s.Constraint(0,s.infinity()) for c,k in enumerate(rules.TYPES[:-1]) if k=='W' for i in range(1,4)}
pots={c:s.Constraint(0,s.infinity()) for c,k in enumerate(rules.TYPES[:-1]) if k=='P'}
budget=s.Constraint(0,800)
obj=s.Objective();obj.SetMaximization()
for n in nodes:
 p,d,i=n
 # Open endpoint; start counts are exact.
 sink=s.NumVar(0,2,'end');flow[n].SetCoefficient(sink,1)
 flow[n].SetBounds(int(n in [(1,0,0),(7,0,0)]),int(n in [(1,0,0),(7,0,0)]))
 for a in range(6):
  np,nd,c=rules.movement(p,d,a)
  options=[(i,None)]
  if a==5:
   k=rules.TYPES[c]
   if k=='O' and i==0:options=[(1,None)]
   elif k=='B' and i==0:options=[(2,None)]
   elif k=='X' and i==3:options=[(0,('deliver',c,0))]
   elif k=='P' and i==1:options=[(0,('fill',c,0)),(i,None)]
   elif k=='P' and i==2:options=[(3,('soup',c,0)),(i,None)]
   elif k=='W':options=([(0,('drop',c,i))] if i else [(v,('take',c,v)) for v in range(1,4)])
  for ni,event in options:
   dest=(np,nd,ni)
   if dest==n and event is None:continue
   v=s.NumVar(0,800,'edge')
   flow[n].SetCoefficient(v,1);flow[dest].SetCoefficient(v,-1)
   budget.SetCoefficient(v,1)
   if event:
    what,c,item=event
    if what=='deliver':obj.SetCoefficient(v,1)
    elif what=='fill':pots[c].SetCoefficient(v,1)
    elif what=='soup':pots[c].SetCoefficient(v,-3)
    else:resources[c,item].SetCoefficient(v,1 if what=='drop' else -1)
# Collision-blocked movement still turns the chef. Permit it freely here.
for n in nodes:
 p,d,i=n
 for a in range(4):
  np,nd,_=rules.movement(p,d,a)
  dest=(p,nd,i)
  if np==p or dest==n:continue
  v=s.NumVar(0,800,'blocked_turn')
  flow[n].SetCoefficient(v,1);flow[dest].SetCoefficient(v,-1)
  budget.SetCoefficient(v,1)
status=s.Solve()
assert status == pywraplp.Solver.OPTIMAL, status
print('LP status',status,'relaxed deliveries upper bound',obj.Value())
