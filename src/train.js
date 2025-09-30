import { pickPlayers } from "./sampling.js";
import { runEpisode } from "./match.js";
import {
  ensureAgent,
  getAgentVersion,
  setAgentVersion,
  clearReplay
} from "./agent.js";
import {
  EPSILON_START,
  EPSILON_END,
  EPSILON_DECAY_STEPS,
  LOCAL_BATCH_SIZE,
  LOCAL_UPDATE_STEPS,
  TRAIN_EPISODES_PER_LOOP
} from "./config.js";
import { fetchModel, uploadDelta } from "./api.js";

function epsilonForStep(step) {
  if (step >= EPSILON_DECAY_STEPS) return EPSILON_END;
  const progress = step / EPSILON_DECAY_STEPS;
  return EPSILON_START + (EPSILON_END - EPSILON_START) * progress;
}

export class Trainer {
  constructor(statusUpdater = () => {}) {
    this.statusUpdater = statusUpdater;
    this.shouldRun = false;
    this.step = 0;
    this.queued = [];
  }

  async initAgents() {
    for (const who of ["gregory", "fred"]) {
      const model = await fetchModel(who);
      ensureAgent(who).setWeightsFromServer(model);
      setAgentVersion(who, model.version);
    }
  }

  async loop() {
    if (!this.shouldRun) return;
    const epsilon = epsilonForStep(this.step);
    const stats = { episodes: 0, steps: 0 };
    for (let i = 0; i < TRAIN_EPISODES_PER_LOOP; i += 1) {
      const pairing = pickPlayers();
      const result = await runEpisode({ ...pairing, epsilon });
      stats.episodes += 1;
      stats.steps += result.steps;
      for (const [who, transitions] of Object.entries(result.transitions)) {
        const agent = ensureAgent(who);
        transitions.forEach((t) => agent.remember(t));
      }
    }
    await this.updateAgents();
    this.step += 1;
    this.statusUpdater({
      eps: (stats.episodes / Math.max(stats.steps, 1)) * 60,
      avgSteps: stats.steps / Math.max(stats.episodes, 1),
      queued: this.queued.length
    });
    setTimeout(() => this.loop(), 0);
  }

  async updateAgents() {
    for (const who of ["gregory", "fred"]) {
      const agent = ensureAgent(who);
      const batch = agent.sampleBatch(LOCAL_BATCH_SIZE);
      if (!batch) continue;
      const before = agent.snapshotLastLayer();
      for (let step = 0; step < LOCAL_UPDATE_STEPS; step += 1) {
        await agent.trainStep(batch);
        agent.maybeSyncTarget();
      }
      const delta = agent.exportLastLayerDelta(before);
      const version = getAgentVersion(who);
      this.queued.push({ who, baseVersion: version, delta, count: batch.length });
    }
  }

  async flush() {
    const pending = [...this.queued];
    this.queued = [];
    for (const item of pending) {
      try {
        await uploadDelta(item);
      } catch (err) {
        console.error("Upload failed", err);
      }
    }
  }
}
