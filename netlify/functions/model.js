import { getStore } from "@netlify/blobs";
import { OBSERVATION_SIZE, ACTIONS } from "../../src/config.js";

const store = getStore({ name: "rl-data" });

const DEFAULT_SPEC = {
  layers: [
    { name: "dense_0_kernel", shape: [OBSERVATION_SIZE, 128] },
    { name: "dense_0_bias", shape: [128] },
    { name: "dense_1_kernel", shape: [128, 128] },
    { name: "dense_1_bias", shape: [128] },
    { name: "dense_2_kernel", shape: [128, ACTIONS.length] },
    { name: "dense_2_bias", shape: [ACTIONS.length] }
  ]
};

const INITIAL_SEED = 424242;

export default async function handler(req) {
  const url = new URL(req.url);
  const who = url.searchParams.get("who");
  if (!isValidWho(who)) {
    return jsonResponse({ error: "invalid who" }, 400);
  }
  const key = `models/${who}/latest.json`;
  let text = await store.get(key);
  if (!text) {
    const initModel = makeInitialModel(DEFAULT_SPEC, INITIAL_SEED);
    text = JSON.stringify(initModel);
    await store.set(key, text, { contentType: "application/json" });
  }
  return new Response(text, { headers: { "content-type": "application/json" } });
}

function isValidWho(who) {
  return who === "gregory" || who === "fred";
}

function makeInitialModel(spec, seed) {
  const rng = mulberry32(seed);
  const weights = {};
  for (const layer of spec.layers) {
    weights[layer.name] = randomArray(layer.shape, rng, 0.05);
  }
  return { version: 0, spec, weights };
}

function randomArray(shape, rng, scale) {
  const count = shape.reduce((a, b) => a * b, 1);
  const arr = new Array(count);
  for (let i = 0; i < count; i += 1) {
    arr[i] = (rng() * 2 - 1) * scale;
  }
  return arr;
}

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  });
}
