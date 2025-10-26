import { getStore } from "@netlify/blobs";

const store = getStore({ name: "rl-data" });
const STATS_KEY = "stats/training.json";

export default async function handler(req) {
  if (req.method !== "GET") {
    return new Response("Method not allowed", { status: 405 });
  }
  const stats = await readStats();
  return jsonResponse(stats);
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
    for (const who of ["gregory", "fred"]) {
      const value = Number(stats.perAgent[who]);
      if (Number.isFinite(value)) {
        result.perAgent[who] = value;
      }
    }
  }
  return result;
}

function createEmptyStats() {
  return {
    total: 0,
    perAgent: {
      gregory: 0,
      fred: 0
    }
  };
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  });
}
