import { getStore } from "@netlify/blobs";

const store = getStore({ name: "rl-data" });
const STATS_KEY = "stats/training.json";
const AGENTS = ["gregory", "fred"];

export default async function handler(req) {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }
  let body;
  try {
    body = await req.json();
  } catch (err) {
    return jsonResponse({ error: "invalid json" }, 400);
  }
  const { who, baseVersion, count, delta } = body || {};
  if (!isValidWho(who) || !Number.isInteger(baseVersion) || !Number.isInteger(count) || count <= 0 || !delta) {
    return jsonResponse({ error: "invalid payload" }, 400);
  }
  const modelTxt = await store.get(`models/${who}/latest.json`);
  if (!modelTxt) {
    return jsonResponse({ error: "model missing" }, 400);
  }
  const model = JSON.parse(modelTxt);
  if (baseVersion !== model.version) {
    return jsonResponse({ error: "stale version" }, 409);
  }
  const valid = validateDelta(delta, model.spec);
  if (!valid.ok) {
    return jsonResponse({ error: valid.error }, 400);
  }
  const key = `updates/${who}/${crypto.randomUUID()}.json`;
  await store.set(key, JSON.stringify({ baseVersion, count, delta }), {
    contentType: "application/json"
  });
  const stats = await incrementTrainingStats(who);
  return jsonResponse({ ok: true, stats });
}

function isValidWho(who) {
  return AGENTS.includes(who);
}

function validateDelta(delta, spec) {
  for (const layer of spec.layers) {
    if (!delta[layer.name]) continue;
    const expected = layer.shape.reduce((a, b) => a * b, 1);
    if (!Array.isArray(delta[layer.name]) || delta[layer.name].length !== expected) {
      return { ok: false, error: `invalid delta for ${layer.name}` };
    }
  }
  return { ok: true };
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  });
}

async function incrementTrainingStats(who) {
  const stats = await readStats();
  stats.total += 1;
  if (!stats.perAgent[who]) {
    stats.perAgent[who] = 0;
  }
  stats.perAgent[who] += 1;
  await store.set(STATS_KEY, JSON.stringify(stats), {
    contentType: "application/json"
  });
  return stats;
}

async function readStats() {
  try {
    const text = await store.get(STATS_KEY);
    if (!text) return createEmptyStats();
    const parsed = JSON.parse(text);
    return normalizeStats(parsed);
  } catch (err) {
    console.error("Failed to read training stats", err);
    return createEmptyStats();
  }
}

function normalizeStats(stats) {
  const result = createEmptyStats();
  if (stats && Number.isFinite(stats.total)) {
    result.total = stats.total;
  }
  if (stats && typeof stats.perAgent === "object") {
    for (const who of AGENTS) {
      const value = Number(stats.perAgent[who]);
      if (Number.isFinite(value)) {
        result.perAgent[who] = value;
      }
    }
  }
  return result;
}

function createEmptyStats() {
  const perAgent = {};
  for (const who of AGENTS) {
    perAgent[who] = 0;
  }
  return { total: 0, perAgent };
}
