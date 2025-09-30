import { getStore } from "@netlify/blobs";

const store = getStore({ name: "rl-data" });
const LR = 0.1;
const MAX_UPDATES = 200;

export default async function handler() {
  for (const who of ["gregory", "fred"]) {
    await aggregateFor(who);
  }
  return new Response("ok");
}

async function aggregateFor(who) {
  const modelKey = `models/${who}/latest.json`;
  const modelTxt = await store.get(modelKey);
  if (!modelTxt) return;
  const model = JSON.parse(modelTxt);
  const list = await store.list({ prefix: `updates/${who}/`, limit: MAX_UPDATES });
  if (!list.blobs || !list.blobs.length) return;
  const acc = {};
  let total = 0;
  for (const blob of list.blobs) {
    const txt = await store.get(blob.key);
    if (!txt) continue;
    const update = JSON.parse(txt);
    if (update.baseVersion !== model.version) {
      continue;
    }
    total += update.count;
    for (const [name, arr] of Object.entries(update.delta)) {
      if (!acc[name]) {
        acc[name] = new Float32Array(arr.length);
      }
      for (let i = 0; i < arr.length; i += 1) {
        acc[name][i] += arr[i] * update.count;
      }
    }
  }
  if (!total) return;
  for (const [name, avg] of Object.entries(acc)) {
    for (let i = 0; i < avg.length; i += 1) {
      avg[i] /= total;
    }
    const target = model.weights[name];
    if (!target) continue;
    for (let i = 0; i < target.length; i += 1) {
      target[i] += avg[i] * LR;
    }
  }
  model.version += 1;
  await store.set(modelKey, JSON.stringify(model), { contentType: "application/json" });
  for (const blob of list.blobs) {
    await store.delete(blob.key);
  }
}
