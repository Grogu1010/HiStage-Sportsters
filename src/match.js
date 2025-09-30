import { FootballEnv } from "./env.js";
import { ensureAgent } from "./agent.js";
import { REWARD_WIN, REWARD_LOSS, REWARD_DRAW, ACTIONS } from "./config.js";

export async function runEpisode({ p1Who, p2Who, epsilon }, opts = {}) {
  const { render = false, renderer = null } = opts;
  const env = new FootballEnv();
  let obs = env.reset();
  const agent1 = ensureAgent(p1Who);
  const agent2 = ensureAgent(p2Who);
  const transitions = {
    [p1Who]: [],
    [p2Who]: []
  };
  let done = false;
  while (!done) {
    const a1 = agent1.epsilonGreedy(obs, epsilon);
    const a2 = agent2.epsilonGreedy(obs, epsilon);
    const prevObs = obs;
    const result = env.step(a1, a2);
    obs = result.observation;
    done = result.done;
    if (render && renderer) {
      renderer(env);
      await waitForNextFrame();
    }
    if (done) {
      let rewards = {};
      if (result.winner === 1) {
        rewards = { [p1Who]: REWARD_WIN, [p2Who]: REWARD_LOSS };
      } else if (result.winner === 2) {
        rewards = { [p1Who]: REWARD_LOSS, [p2Who]: REWARD_WIN };
      } else {
        rewards = { [p1Who]: REWARD_DRAW, [p2Who]: REWARD_DRAW };
      }
      transitions[p1Who].push({
        state: prevObs,
        action: a1,
        nextState: obs,
        reward: rewards[p1Who],
        done: true
      });
      transitions[p2Who].push({
        state: prevObs,
        action: a2,
        nextState: obs,
        reward: rewards[p2Who],
        done: true
      });
    } else {
      transitions[p1Who].push({
        state: prevObs,
        action: a1,
        nextState: obs,
        reward: 0,
        done: false
      });
      transitions[p2Who].push({
        state: prevObs,
        action: a2,
        nextState: obs,
        reward: 0,
        done: false
      });
    }
  }
  return { transitions, steps: env.stepCount };
}

function waitForNextFrame() {
  if (typeof requestAnimationFrame === "function") {
    return new Promise((resolve) => requestAnimationFrame(resolve));
  }
  return new Promise((resolve) => setTimeout(resolve, 16));
}
