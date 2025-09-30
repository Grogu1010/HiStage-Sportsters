import { Trainer } from "./train.js";
import { fetchModel } from "./api.js";
import { ensureAgent, setAgentVersion } from "./agent.js";

const canvas = document.getElementById("pitch");
const ctx = canvas.getContext("2d");
const toggleBtn = document.getElementById("train-toggle");
const watchBtn = document.getElementById("watch-once");
const watchTrainingBtn = document.getElementById("watch-training");
const trainingIndicator = document.getElementById("training-indicator");
const trainingIndicatorText = document.getElementById("training-indicator-text");
const epsEl = document.getElementById("eps");
const avgStepsEl = document.getElementById("avg-steps");
const queuedEl = document.getElementById("queued-updates");
const versionEls = {
  gregory: document.getElementById("gregory-version"),
  fred: document.getElementById("fred-version")
};

const trainer = new Trainer(updateStatus, drawScene);
let renderHandle = null;
let watchingTraining = false;

(async () => {
  await trainer.initAgents();
  await refreshVersions();
  setInterval(() => trainer.flush(), 15000);
  updateTrainingUI();
})();

toggleBtn.addEventListener("click", async () => {
  trainer.shouldRun = !trainer.shouldRun;
  if (trainer.shouldRun) {
    trainer.loop();
  } else if (watchingTraining) {
    watchingTraining = false;
    trainer.setSpectate(false);
    updateWatchTrainingBtn();
  }
  updateTrainingUI();
});

watchTrainingBtn.addEventListener("click", () => {
  watchingTraining = !watchingTraining;
  trainer.setSpectate(watchingTraining);
  updateWatchTrainingBtn();
  if (watchingTraining && !trainer.shouldRun) {
    trainer.shouldRun = true;
    trainer.loop();
    updateTrainingUI();
  }
});

watchBtn.addEventListener("click", async () => {
  if (renderHandle) cancelAnimationFrame(renderHandle);
  const { runEpisode } = await import("./match.js");
  const { pickPlayers } = await import("./sampling.js");
  const pairing = pickPlayers();
  const epsilon = 0.01;
  await runEpisode({ ...pairing, epsilon }, { render: true, renderer: drawScene });
});

async function refreshVersions() {
  for (const who of ["gregory", "fred"]) {
    try {
      const model = await fetchModel(who);
      ensureAgent(who).setWeightsFromServer(model);
      setAgentVersion(who, model.version);
      versionEls[who].textContent = model.version;
    } catch (err) {
      console.error("Failed to refresh", who, err);
    }
  }
  setTimeout(refreshVersions, 45000);
}

function updateStatus({ eps, avgSteps, queued }) {
  epsEl.textContent = eps.toFixed(2);
  avgStepsEl.textContent = avgSteps.toFixed(0);
  queuedEl.textContent = queued;
}

function updateTrainingUI() {
  toggleBtn.textContent = trainer.shouldRun ? "Stop Training" : "Start Training";
  toggleBtn.classList.toggle("active", trainer.shouldRun);
  trainingIndicator.dataset.state = trainer.shouldRun ? "active" : "idle";
  trainingIndicatorText.textContent = trainer.shouldRun
    ? "Training active"
    : "Training paused";
}

function updateWatchTrainingBtn() {
  watchTrainingBtn.textContent = watchingTraining
    ? "Stop Watching Training"
    : "Watch Training";
  watchTrainingBtn.classList.toggle("active", watchingTraining);
}

function drawScene(env) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#133b22";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,0.4)";
  ctx.lineWidth = 4;
  ctx.strokeRect(20, 20, width - 40, height - 40);
  ctx.beginPath();
  ctx.moveTo(width / 2, 20);
  ctx.lineTo(width / 2, height - 20);
  ctx.stroke();
  drawCircle(width / 2, height / 2, 40, "rgba(255,255,255,0.3)");
  drawBall(env.ball);
  drawPlayer(env.players[0], "#5dade2");
  drawPlayer(env.players[1], "#f1948a");
  env.cones.forEach((cone) => drawCone(cone));
}

function drawBall(ball) {
  const pos = toCanvas(ball.x, ball.y);
  ctx.fillStyle = "#f8f1e1";
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2);
  ctx.fill();
}

function drawPlayer(player, color) {
  const pos = toCanvas(player.x, player.y);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  ctx.lineTo(pos.x + Math.cos(player.facing) * 18, pos.y + Math.sin(player.facing) * 18);
  ctx.stroke();
}

function drawCone(cone) {
  const pos = toCanvas(cone.x, cone.y);
  ctx.save();
  ctx.translate(pos.x, pos.y);
  ctx.rotate(cone.angle || 0);
  ctx.fillStyle = "#ff7043";
  ctx.beginPath();
  ctx.moveTo(0, -10);
  ctx.lineTo(8, 10);
  ctx.lineTo(-8, 10);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawCircle(x, y, radius, style) {
  ctx.strokeStyle = style;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.stroke();
}

function toCanvas(x, y) {
  const px = ((x + 30) / 60) * canvas.width;
  const py = ((y + 18) / 36) * canvas.height;
  return { x: px, y: py };
}
