import {
  PHYSICS,
  computeCartPoleTransition,
  stepCartPole
} from '../src/lib/physics.js';

const assert=(ok,msg)=>{if(!ok) throw new Error(msg);};
const close=(a,b,eps=1e-12)=>Math.abs(a-b)<=eps;

const state={x:0.12,xDot:-0.31,theta:0.07,thetaDot:0.42};
const policy=6.4;
const disturbance=-2.25;
const t=computeCartPoleTransition(state,policy,PHYSICS.tau,disturbance);
const legacy=stepCartPole(state,policy,PHYSICS.tau,disturbance);

for(const key of ['x','xDot','theta','thetaDot']){
  assert(close(t.nextState[key],legacy[key]),'detailed transition diverges from stepCartPole at '+key);
}
assert(close(t.totalForce,t.control+t.disturbance),'total force sum mismatch');
assert(close(t.nextState.x,state.x+t.dt*state.xDot),'x Euler update mismatch');
assert(close(t.nextState.xDot,state.xDot+t.dt*t.xAcc),'xDot Euler update mismatch');
assert(close(t.nextState.theta,state.theta+t.dt*state.thetaDot),'theta Euler update mismatch');
assert(close(t.nextState.thetaDot,state.thetaDot+t.dt*t.thetaAcc),'thetaDot Euler update mismatch');
assert(Number.isFinite(t.temp)&&Number.isFinite(t.xAcc)&&Number.isFinite(t.thetaAcc),'non-finite dynamics intermediate');

const clipped=computeCartPoleTransition(state,25,PHYSICS.tau,3);
assert(close(clipped.control,PHYSICS.forceMag),'policy clamp mismatch');
assert(close(clipped.totalForce,PHYSICS.forceMag+3),'clipped total force mismatch');

console.log('dynamics transition check passed',{
  totalForce:t.totalForce,
  xAcc:t.xAcc,
  thetaAcc:t.thetaAcc,
  nextState:t.nextState
});
