"""
pl15_multisensor.py
CRT-style multi-sensor PL-15 training system
CuPy ≥ 3.12 only • dual Torch/JAX back-ends • real-time visualization
"""

import math, sys, time, random, statistics
from math import prod, log, radians, sin, hypot, atan2, degrees
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

try:
    import cupy as cp  # CuPy ≥ 3.12
    assert tuple(map(int, cp.__version__.split('.')[:2])) >= (3, 12)
except (ImportError, AssertionError):
    cp = None
try:
    import torch
except ImportError:
    torch = None
try:
    import jax, jax.numpy as jnp
except ImportError:
    jax = None

# ───────────────────────── original utilities ──────────────────────────
def engineering_thought_modular_composition(residues, moduli):
    M = prod(moduli)
    if len(residues) == 2:
        mi, mf = residues; isp, g = moduli
        return 0 if mi <= mf else isp*g*log(mi/mf)+0.1*(isp*g*log(mi/mf))**.8+M
    if len(residues) == 3:
        mi, thrust, isp = residues; mf, burn_time, g = moduli
        if mi <= mf: return 0
        dv = isp*g*log(mi/mf); avg_acc = thrust/((mi+mf)/2)
        return dv+0.2*avg_acc**.7+burn_time**.3+M
    return 0

def hypersonic_drag_coefficient(M):
    return 0.5 if M<1 else 0.3 if M<3 else 0.28 if M<5 else 0.25 if M<10 else 0.23

def _rho(h):
    return 1.225*(1-2.25577e-5*h)**4.256 if h<11000 else \
           0.36391*(1-2.25577e-5*11000)**4.256*np.exp(-(h-11000)/6341.62) if h<20000 else \
           0.08803*np.exp(-(h-20000)/6341.62)

def _a(h): return 340.29-0.0065*h if h<11000 else 295.07

def approximate_hypersonic_trajectory(m, T, isp, tb, dt=.05):
    g, A, dv, v, s, h = 9.80665, .3, isp*9.80665*log(m/(m-T*tb/(isp*9.80665))), 0, 0, 0
    ig = isp*g; n=int(tb/dt)
    for _ in range(n):
        m=max(1e-6, m-T/ig*dt)
        rho, a = _rho(h), _a(h)
        cd = hypersonic_drag_coefficient(v/a); D=.5*rho*v*v*A*cd
        a_net=(T-D)/m-(g if h<20000 else 0); v+=a_net*dt; s+=v*dt
        h+=max(0., v*.05*(1-D/(T+1e-9)))
    return v, s, h, dv

# ───────────────────────── sensor modules ──────────────────────────────
class RadarTorch:
    def __init__(self, mode='pulse', aesa=True): self.mode,self.aesa,self.max_r=mode,aesa,1e5
    def sim(self,t,e):
        tx,ty,tvx,tvy,rcs=(t[k] for k in('x','y','vx','vy','rcs')); r=hypot(tx,ty)
        if r>self.max_r: return None
        dop=(tx*tvx+ty*tvy)/(r+1e-6); p=rcs/r**4*(2 if self.aesa else 1)
        n=(cp.random.randn()*1e-6).item() if cp else random.gauss(0,1e-6)
        if e['jam']: n+=e['jp']*random.uniform(.5,1)
        return {'r':r+n*r,'v':dop+n*dop,'c':.5 if e['jam'] else 1}
    def proc(self,m): return None if m is None else {'range':m['r'],'vel':m['v'],'conf':m['c']}

class RadarJAX:
    def __init__(self, mode='pulse', aesa=True): self.mode,self.aesa,self.max_r=mode,aesa,1e5
    @staticmethod
    @jax.jit
    def _f(tx,ty,tvx,tvy,rcs,aesa,j,jp):
        r=jnp.sqrt(tx*tx+ty*ty); dop=(tx*tvx+ty*tvy)/(r+1e-6)
        n=1e-6*jax.random.normal(jax.random.PRNGKey(0),()); n=jnp.where(j,n+jp*.75,n)
        rc=jnp.where(aesa,rcs*2,rcs); return r+n*r,dop+n*dop,jnp.where(j,.5,1)
    def detect(self,t,e):
        tx,ty,tvx,tvy,rcs=(t[k] for k in('x','y','vx','vy','rcs'))
        if tx*tx+ty*ty>self.max_r**2: return None
        r,v,c=self._f(tx,ty,tvx,tvy,rcs,self.aesa,e['jam'],e['jp'])
        return {'range':float(r),'vel':float(v),'conf':float(c)}

class IRTorch:
    def __init__(self,fov=30,rng=2e4): self.fov,self.max_r=fov,rng
    def sim(self,t,e):
        tx,ty,ir=t['x'],t['y'],t['ir']; r=hypot(tx,ty)
        if r>self.max_r: return None
        ang=degrees(atan2(ty,tx)); 
        if abs(ang)>self.fov: return None
        flare=e['flare'] and e['fi']>ir*1.2/r**2
        return {'a':ang+random.uniform(-.1,.1),'c':.4 if flare else 1}
    def proc(self,m): return None if m is None else {'ang':m['a'],'conf':m['c']}

class IRJAX:
    def __init__(self,fov=30,rng=2e4): self.fov,self.max_r=fov,rng
    def detect(self,t,e):
        tx,ty,ir=t['x'],t['y'],t['ir']; r=math.hypot(tx,ty)
        if r>self.max_r: return None
        ang=math.degrees(math.atan2(ty,tx)); 
        if abs(ang)>self.fov: return None
        flare=e['flare'] and e['fi']>ir*1.2/r**2
        return {'ang':ang,'conf':.4 if flare else 1}

class RF:
    def detect(self,t,e): 
        return {'jam':e['jam'],'bearing':degrees(atan2(t['y'],t['x'])),'conf':1}

class Motion:
    def predict(self,est,dt): 
        return {'x':est['x']+est['vx']*dt,'y':est['y']+est['vy']*dt,'vx':est['vx'],'vy':est['vy']}

# ───────────────────────── fusion / health / decision ─────────────────
class Fusion:
    def __init__(self): self.est={'x':0,'y':0,'vx':0,'vy':0}
    def update(self,r,ir,p,wt=1):
        sx,sy,svx,svy,w=0,0,0,0,0
        if p: sx+=p['x']; sy+=p['y']; svx+=p['vx']; svy+=p['vy']; w+=wt
        if r: sx+=r['range']*r['conf']; sy+=0; svx+=-r['vel']*r['conf']; w+=r['conf']
        if ir:
            rng=hypot(sx/w,sy/w) if w else 5e3
            sx+=ir['conf']*rng*math.cos(math.radians(ir['ang']))
            sy+=ir['conf']*rng*math.sin(math.radians(ir['ang'])); w+=ir['conf']
        if w: self.est={'x':sx/w,'y':sy/w,'vx':svx/w,'vy':svy/w}
        return self.est

class Health:
    def assess(self,r,ir,rf):
        return {
            'rad':'jam' if rf['jam'] else ('bad' if not r else 'ok'),
            'ir':'bad' if ir and ir['conf']<.5 else ('miss' if not ir else 'ok')
        }

class Decision:
    def __init__(self): self.primary='radar'
    def step(self,h):
        if h['rad']=='jam' and h['ir']=='ok': self.primary='ir'
        elif h['ir']=='bad' and h['rad']=='ok': self.primary='radar'
        return self.primary

# ───────────────────────── simulation harness ─────────────────────────
class Sim:
    def __init__(self,visual=False,deploy=False):
        self.radar=RadarJAX() if deploy and jax else RadarTorch()
        self.ir   =IRJAX()    if deploy and jax else IRTorch()
        self.rf=RF(); self.motion=Motion()
        self.fus=Fusion(); self.hlth=Health(); self.dec=Decision()
        self.t={'x':3e4,'y':0,'vx':-300,'vy':0,'rcs':5,'ir':1}
        self.e={'jam':False,'jp':0,'flare':False,'fi':0}
        self.tnow=0; self.vis=visual
        if visual: self._init_plot()
    def _init_plot(self):
        self.fig,self.ax=plt.subplots(); self.ax.set_xlim(-500,30000); self.ax.set_ylim(-1e4,1e4)
        self.tp,=self.ax.plot([],[],'ro',label='target'); self.ep,=self.ax.plot([],[],'bx',label='est')
        self.ax.legend(); plt.ion(); plt.show()
    def _update_plot(self):
        self.tp.set_data(self.t['x'],self.t['y']); self.ep.set_data(self.fus.est['x'],self.fus.est['y'])
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
    def step(self,dt=.5):
        self.tnow+=dt
        if self.tnow>=10 and not self.e['jam']: self.e.update(jam=True,jp=1e-3)
        if self.tnow>=15 and not self.e['flare']:
            self.e.update(flare=True,fi=5); self.t.update(vy=200,vx=-250)
        self.t['x']+=self.t['vx']*dt; self.t['y']+=self.t['vy']*dt
        r=self.radar.proc(self.radar.sim(self.t,self.e)) if isinstance(self.radar,RadarTorch) else self.radar.detect(self.t,self.e)
        ir=self.ir.proc(self.ir.sim(self.t,self.e)) if isinstance(self.ir,IRTorch) else self.ir.detect(self.t,self.e)
        rf=self.rf.detect(self.t,self.e); pred=self.motion.predict(self.fus.est,dt)
        est=self.fus.update(r,ir,pred); h=self.hlth.assess(r,ir,rf); primary=self.dec.step(h)
        if self.vis: self._update_plot()
        return est,h,primary
def run():
    s=Sim(visual=True); for _ in range(60): s.step(); plt.ioff(); plt.show()
def batch(n=100):
    errs=[]; t0=time.time()
    for _ in range(n):
        s=Sim(); [s.step() for _ in range(60)]
        est=s.fus.est; errs.append(hypot(est['x']-s.t['x'],est['y']-s.t['y']))
    print(f"avg {statistics.mean(errs):.1f}m  <100m {sum(e<100 for e in errs)/n:%}  time {time.time()-t0:.1f}s")
    plt.hist(errs,bins=20); plt.xlabel('final error m'); plt.ylabel('count'); plt.title('Batch error distribution'); plt.show()

# ───────────────────────── CLI ────────────────────────────────────────
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='batch': batch(200)
    else: run()
