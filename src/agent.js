import * as config from "./config.js";

const agents = new Map();

class AgentModel {
  constructor(who) {
    this.who = who;
    this.online = buildNetwork();
    this.target = buildNetwork();
    this.optimizer = tf.train.adam(config.LEARNING_RATE);
    this.version = -1;
    this.steps = 0;
    this.replay = [];
  }

  setWeightsFromServer(modelPayload) {
    const { weights, version } = modelPayload;
    this.version = version;
    assignWeights(this.online, weights);
    assignWeights(this.target, weights);
    if (this.optimizer) {
      this.optimizer.dispose();
    }
    this.optimizer = tf.train.adam(config.LEARNING_RATE);
  }

  predict(obs, useTarget = false) {
    const model = useTarget ? this.target : this.online;
    return tf.tidy(() => {
      const input = tf.tensor2d([obs]);
      const logits = model.predict(input);
      return logits.squeeze();
    });
  }

  epsilonGreedy(obs, epsilon) {
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * config.ACTIONS.length);
    }
    const q = this.predict(obs);
    const action = q.argMax().arraySync();
    q.dispose();
    return action;
  }

  remember(transition) {
    this.replay.push(transition);
    if (this.replay.length > config.MAX_REPLAY_SIZE) {
      this.replay.splice(0, this.replay.length - config.MAX_REPLAY_SIZE);
    }
  }

  sampleBatch(size) {
    if (this.replay.length < size) return null;
    const batch = [];
    for (let i = 0; i < size; i += 1) {
      const idx = Math.floor(Math.random() * this.replay.length);
      batch.push(this.replay[idx]);
    }
    return batch;
  }

  async trainStep(batch) {
    const loss = this.optimizer.minimize(() => {
      const states = tf.tensor2d(batch.map((b) => b.state));
      const nextStates = tf.tensor2d(batch.map((b) => b.nextState));
      const actions = tf.tensor1d(batch.map((b) => b.action), "int32");
      const rewards = tf.tensor1d(batch.map((b) => b.reward));
      const terminals = tf.tensor1d(batch.map((b) => (b.done ? 0 : 1)));
      const actionMask = tf.oneHot(actions, config.ACTIONS.length);
      const qValues = this.online.apply(states);
      const pred = qValues.mul(actionMask).sum(-1);
      const nextOnline = this.online.apply(nextStates);
      const nextActions = nextOnline.argMax(-1);
      const nextActionMask = tf.oneHot(nextActions, config.ACTIONS.length);
      const nextTarget = this.target.apply(nextStates);
      const nextMaxQ = nextTarget.mul(nextActionMask).sum(-1);
      const targets = rewards.add(nextMaxQ.mul(terminals).mul(config.GAMMA));
      return tf.losses.huberLoss(targets, pred);
    }, true);
    if (loss) loss.dispose();
  }

  maybeSyncTarget() {
    this.steps += 1;
    if (this.steps % config.TARGET_SYNC_INTERVAL === 0) {
      const weights = this.online.getWeights();
      this.target.setWeights(weights);
      weights.forEach((w) => w.dispose());
    }
  }

  exportLastLayerDelta(beforeWeights) {
    const after = this.online.getWeights();
    const delta = {};
    const names = this.online.layers[this.online.layers.length - 1].weights.map((w) => w.originalName);
    names.forEach((name, idx) => {
      const afterVals = after[after.length - names.length + idx].flatten().arraySync();
      const beforeVals = beforeWeights[idx];
      delta[name] = afterVals.map((v, i) => v - beforeVals[i]);
    });
    after.forEach((w) => w.dispose());
    return delta;
  }

  snapshotLastLayer() {
    const layer = this.online.layers[this.online.layers.length - 1];
    return layer.weights.map((w) => Array.from(w.read().dataSync()));
  }
}

function buildNetwork() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [config.OBSERVATION_SIZE],
      units: 128,
      activation: "relu",
      name: "dense_0"
    })
  );
  model.add(tf.layers.dense({ units: 128, activation: "relu", name: "dense_1" }));
  model.add(tf.layers.dense({ units: config.ACTIONS.length, activation: "linear", name: "dense_2" }));
  model.compile({ optimizer: tf.train.adam(0.0005), loss: "meanSquaredError" });
  return model;
}

function assignWeights(model, weightsObj) {
  const weights = [];
  model.weights.forEach((w) => {
    const arr = weightsObj[w.originalName];
    if (!arr) return;
    const shape = w.shape;
    const tensor = tf.tensor(arr, shape);
    weights.push(tensor);
  });
  model.setWeights(weights);
  weights.forEach((w) => w.dispose());
}

export function ensureAgent(who) {
  if (!agents.has(who)) {
    agents.set(who, new AgentModel(who));
  }
  return agents.get(who);
}

export function applyDelta(who, delta) {
  const agent = ensureAgent(who);
  const layer = agent.online.layers[agent.online.layers.length - 1];
  layer.weights.forEach((variable) => {
    const name = variable.originalName;
    const update = delta[name];
    if (!update) return;
    const current = variable.read().arraySync();
    const flat = flatten(current);
    for (let i = 0; i < update.length; i += 1) {
      flat[i] += update[i];
    }
    const tensor = tf.tensor(flat, variable.shape);
    variable.write(tensor);
    tensor.dispose();
  });
  agent.target.setWeights(agent.online.getWeights());
}

function flatten(values) {
  if (Array.isArray(values[0])) {
    return values.flat(Infinity);
  }
  return [...values];
}

export function getAgentVersion(who) {
  const agent = agents.get(who);
  return agent ? agent.version : -1;
}

export function setAgentVersion(who, version) {
  const agent = ensureAgent(who);
  agent.version = version;
}

export function getReplay(who) {
  return ensureAgent(who).replay;
}

export function clearReplay(who) {
  ensureAgent(who).replay = [];
}
