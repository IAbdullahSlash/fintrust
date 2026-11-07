"""Research-only binary, single-task governance core. No broker connectivity."""
from dataclasses import dataclass, field, asdict
import math, json, hashlib

@dataclass(frozen=True)
class Config:
    prior_mass: float = 20
    decay: float = .9
    temperature: float = .2
    window: int = 50
    minimum: int = 200
    consecutive: int = 3
    bad: float = .05
    good: float = 0
    ttl: int = 1
    auto_clear_statistical: bool = True  # two-tier rule: statistical quarantines self-clear; integrity faults need clear()
    def __post_init__(self):
        if not (self.prior_mass>0 and 0<self.decay<=1 and self.temperature>0 and self.window>0 and self.minimum>0 and self.consecutive>0 and self.ttl>0):
            raise ValueError('invalid configuration')

@dataclass
class Agent:
    state: str = 'SHADOW'
    loss_sum: float = 5
    mass: float = 20
    count: int = 0
    since: int = 0
    good: int = 0
    bad: int = 0
    window: list = field(default_factory=list)
    cleared: bool = True
    @property
    def loss(self): return self.loss_sum/self.mass if self.mass>1e-12 else .25

class Audit:
    def __init__(self): self.records=[]
    @staticmethod
    def digest(previous, payload):
        raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
        return hashlib.sha256((previous+'\n'+raw).encode()).hexdigest()
    def append(self,payload):
        payload=json.loads(json.dumps(payload,allow_nan=False))
        previous=self.records[-1]['hash'] if self.records else '0'*64
        self.records.append(dict(previous=previous,payload=payload,hash=self.digest(previous,payload)))
    def verify(self,checkpoint=None):
        previous='0'*64
        for r in self.records:
            if r['previous']!=previous or r['hash']!=self.digest(previous,r['payload']):return False
            previous=r['hash']
        return checkpoint is None or checkpoint==(len(self.records),previous)
    def checkpoint(self):return (len(self.records),self.records[-1]['hash'] if self.records else '0'*64)

class Governor:
    def __init__(self,config=Config()):
        self.c=config;self.agents={};self.pending={};self.resolved=set();self.audit=Audit();self.now=-1;self.version=0;self.halted=False;self.committed=None
    def register(self,agent):
        if agent in self.agents:raise ValueError('duplicate identity')
        self.agents[agent]=Agent(loss_sum=self.c.prior_mass*.25,mass=self.c.prior_mass)
    def advance(self,t):
        if t<self.now:raise ValueError('clock regression')
        if self.now>=0:
            factor=self.c.decay**(t-self.now)
            for a in self.agents.values():a.mass*=factor;a.loss_sum*=factor
        self.now=t
    def forecast(self,key,agent,p,decision,horizon,expiry):
        if key in self.pending or key in self.resolved:return False
        if agent not in self.agents or not math.isfinite(p) or not 0<=p<=1 or horizon<=0 or decision>self.now or expiry<=self.now:
            self.audit.append(dict(kind='reject',key=key,time=self.now));return False
        self.pending[key]=dict(agent=agent,p=p,end=decision+horizon)
        return True
    def outcome(self,key,y,available):
        if key not in self.pending:return False
        f=self.pending[key]
        if y not in (0,1):raise ValueError('binary outcome required')
        if available<f['end']:raise ValueError('outcome predates horizon')
        if available>self.now:return False
        a=self.agents[f['agent']];loss=(f['p']-y)**2
        a.mass+=1;a.loss_sum+=loss;a.count+=1;a.since+=1;a.window.append(loss-.25)
        if len(a.window)==self.c.window:
            delta=sum(a.window)/len(a.window);a.window=[]
            a.good=a.good+1 if delta<self.c.good else 0
            a.bad=a.bad+1 if delta>self.c.bad else 0
            old=a.state
            if a.state=='ACTIVE' and a.bad>=self.c.consecutive:a.state='QUARANTINED';a.cleared=self.c.auto_clear_statistical
            elif a.state=='PROBATION' and a.bad:a.state='QUARANTINED';a.cleared=self.c.auto_clear_statistical
            elif a.state=='SHADOW' and a.count>=self.c.minimum and a.good>=self.c.consecutive:a.state='ACTIVE'
            elif a.state=='QUARANTINED' and a.cleared and a.since>=self.c.minimum and a.good>=self.c.consecutive:a.state='PROBATION'
            elif a.state=='PROBATION' and a.good>=self.c.consecutive:a.state='ACTIVE'
            if old!=a.state:
                a.since=a.good=a.bad=0
                self.audit.append(dict(kind='transition',agent=f['agent'],old=old,new=a.state,time=self.now))
        self.resolved.add(key);del self.pending[key]
        self.audit.append(dict(kind='outcome',key=key,time=self.now,available=available,loss=loss))
        return True
    def fault(self,agent):
        a=self.agents[agent];a.state='QUARANTINED';a.cleared=False;a.since=a.good=a.bad=0;a.window=[]
        self.audit.append(dict(kind='fault',agent=agent,time=self.now))
    def clear(self,agent):
        self.agents[agent].cleared=True
        self.audit.append(dict(kind='root_cause_cleared',agent=agent,time=self.now))
    def policy(self,forecasts):
        eligible={k:v for k,v in forecasts.items() if k in self.agents and self.agents[k].state in ('ACTIVE','PROBATION') and math.isfinite(v) and 0<=v<=1}
        caps={k:(.1 if self.agents[k].state=='PROBATION' else 1.) for k in eligible}
        weights={};reason='active'
        if self.halted or not eligible or sum(caps.values())<1-1e-12:reason='halt_or_infeasible'
        else:
            scores={k:(1-self.agents[k].loss)/self.c.temperature for k in eligible};mx=max(scores.values());raw={k:math.exp(v-mx) for k,v in scores.items()}
            remaining=set(raw);budget=1.
            while remaining:
                total=sum(raw[k] for k in remaining)
                trial={k:budget*raw[k]/total for k in remaining}
                clipped={k for k in remaining if trial[k]>caps[k]+1e-12}
                if not clipped:weights.update(trial);break
                for k in clipped:weights[k]=caps[k];budget-=caps[k]
                remaining-=clipped
        self.version+=1
        record=dict(version=self.version,created=self.now,expires=self.now+self.c.ttl,weights=weights,probability=sum(weights[k]*eligible[k] for k in weights) if weights else None,reason=reason)
        self.audit.append(dict(kind='policy',**record));self.committed=json.loads(json.dumps(record));return record
    def authorize(self,policy,new_risk=True):
        return bool(policy==self.committed and not self.halted and policy['version']==self.version and self.now<policy['expires'] and (policy['weights'] or not new_risk))
