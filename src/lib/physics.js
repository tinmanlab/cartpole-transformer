export const PHYSICS = {
  gravity: 9.8,
  massCart: 1.0,
  massPole: 0.1,
  halfPoleLength: 0.5,
  forceMag: 10.0,
  tau: 0.02,
};

export function resetState(angle = 0.045) {
  return { x: 0, xDot: 0, theta: angle, thetaDot: 0 };
}

export function stepCartPole(state, actionForce, dt = PHYSICS.tau) {
  const { gravity, massCart, massPole, halfPoleLength } = PHYSICS;
  const totalMass = massCart + massPole;
  const poleMassLength = massPole * halfPoleLength;
  const force = Math.max(-PHYSICS.forceMag, Math.min(PHYSICS.forceMag, actionForce));
  const { x, xDot, theta, thetaDot } = state;
  const costheta = Math.cos(theta);
  const sintheta = Math.sin(theta);
  const temp = (force + poleMassLength * thetaDot * thetaDot * sintheta) / totalMass;
  const thetaAcc = (gravity * sintheta - costheta * temp) /
    (halfPoleLength * (4 / 3 - massPole * costheta * costheta / totalMass));
  const xAcc = temp - poleMassLength * thetaAcc * costheta / totalMass;

  return {
    x: x + dt * xDot,
    xDot: xDot + dt * xAcc,
    theta: theta + dt * thetaDot,
    thetaDot: thetaDot + dt * thetaAcc,
  };
}

export function terminal(state) {
  return Math.abs(state.x) > 2.4 || Math.abs(state.theta) > 0.38;
}

export function stateArray(s) {
  return [s.x, s.xDot, s.theta, s.thetaDot];
}
