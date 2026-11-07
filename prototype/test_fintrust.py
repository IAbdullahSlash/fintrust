import unittest,copy,math
from fintrust import Governor,Config,Audit
class Tests(unittest.TestCase):
 def setup_g(self):
  g=Governor(Config(window=2,minimum=4,consecutive=2));g.register('a');g.advance(0);return g
 def feed(self,g,p,y,n):
  for _ in range(n):
   t=g.now;k=str(t);g.forecast(k,'a',p,t,1,t+1);g.advance(t+1);g.outcome(k,y,t+1)
 def test_delayed_label(self):
  g=self.setup_g();g.forecast('x','a',.8,0,2,1);before=g.agents['a'].loss;self.assertFalse(g.outcome('x',1,2));self.assertEqual(before,g.agents['a'].loss);g.advance(2);self.assertTrue(g.outcome('x',1,2))
 def test_duplicate_label(self):
  g=self.setup_g();self.feed(g,.8,1,1);n=g.agents['a'].count;self.assertFalse(g.outcome('0',1,1));self.assertEqual(n,g.agents['a'].count)
 def test_invalid_probability(self):
  g=self.setup_g();self.assertFalse(g.forecast('x','a',math.nan,0,1,1))
 def test_expired_forecast(self):
  g=self.setup_g();self.assertFalse(g.forecast('x','a',.8,0,1,0))
 def test_clock_regression(self):
  g=self.setup_g();g.advance(2)
  with self.assertRaises(ValueError):g.advance(1)
 def test_shadow_promotion(self):
  g=self.setup_g();self.feed(g,.9,1,4);self.assertEqual(g.agents['a'].state,'ACTIVE')
 def test_quarantine(self):
  g=self.setup_g();self.feed(g,.9,1,4);self.feed(g,.99,0,4);self.assertEqual(g.agents['a'].state,'QUARANTINED')
 def test_recovery_requires_clearance(self):
  g=self.setup_g();g.fault('a');self.feed(g,.9,1,4);self.assertEqual(g.agents['a'].state,'QUARANTINED');g.clear('a');self.feed(g,.9,1,2);self.assertEqual(g.agents['a'].state,'PROBATION')
 def test_probation_cap(self):
  g=self.setup_g();g.register('b');g.agents['a'].state='PROBATION';g.agents['b'].state='ACTIVE';p=g.policy({'a':.8,'b':.6});self.assertLessEqual(p['weights']['a'],.1);self.assertAlmostEqual(sum(p['weights'].values()),1)
 def test_infeasible_caps(self):
  g=self.setup_g();g.agents['a'].state='PROBATION';self.assertEqual(g.policy({'a':.7})['weights'],{})
 def test_stale_policy(self):
  g=self.setup_g();g.agents['a'].state='ACTIVE';p=g.policy({'a':.7});self.assertTrue(g.authorize(p));g.advance(1);self.assertFalse(g.authorize(p))
 def test_policy_mutation(self):
  g=self.setup_g();g.agents['a'].state='ACTIVE';p=g.policy({'a':.7});p['expires']=999;self.assertFalse(g.authorize(p))
 def test_statistical_quarantine_self_clears(self):
  g=self.setup_g();self.feed(g,.9,1,4);self.feed(g,.99,0,4);self.assertEqual(g.agents['a'].state,'QUARANTINED');self.feed(g,.9,1,4);self.assertEqual(g.agents['a'].state,'PROBATION')
 def test_statistical_quarantine_manual_mode(self):
  g=Governor(Config(window=2,minimum=4,consecutive=2,auto_clear_statistical=False));g.register('a');g.advance(0);self.feed(g,.9,1,4);self.feed(g,.99,0,4);self.feed(g,.9,1,6);self.assertEqual(g.agents['a'].state,'QUARANTINED')
 def test_halt(self):
  g=self.setup_g();g.agents['a'].state='ACTIVE';p=g.policy({'a':.7});g.halted=True;self.assertFalse(g.authorize(p));self.assertEqual(g.policy({'a':.7})['weights'],{})
 def test_audit_mutation(self):
  a=Audit();a.append({'x':1});cp=a.checkpoint();self.assertTrue(a.verify(cp));a.records[0]['payload']['x']=2;self.assertFalse(a.verify(cp))
 def test_audit_truncation(self):
  a=Audit();a.append({'x':1});a.append({'x':2});cp=a.checkpoint();a.records.pop();self.assertTrue(a.verify());self.assertFalse(a.verify(cp))
if __name__=='__main__':unittest.main(verbosity=2)
