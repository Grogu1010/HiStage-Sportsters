import {
  FIELD_WIDTH,
  FIELD_HEIGHT,
  BALL_RADIUS,
  PLAYER_RADIUS,
  CONE_RADIUS,
  STEP_DT,
  FRICTION,
  BALL_FRICTION,
  MAX_STEPS,
  ACTIONS,
  GOAL_HALF_WIDTH
} from "./config.js";

const HALF_WIDTH = FIELD_WIDTH / 2;
const HALF_HEIGHT = FIELD_HEIGHT / 2;

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function vecLength(x, y) {
  return Math.sqrt(x * x + y * y);
}

function normalise(x, y) {
  const len = vecLength(x, y) || 1;
  return [x / len, y / len];
}

export class FootballEnv {
  constructor() {
    this.reset();
  }

  reset(seed = Math.random()) {
    this.stepCount = 0;
    this.ball = {
      x: 0,
      y: 0,
      vx: 0,
      vy: 0
    };
    this.players = [
      {
        x: -FIELD_WIDTH * 0.25,
        y: 0,
        vx: 0,
        vy: 0,
        facing: 0
      },
      {
        x: FIELD_WIDTH * 0.25,
        y: 0,
        vx: 0,
        vy: 0,
        facing: Math.PI
      }
    ];
    this.cones = [
      createCone(-HALF_WIDTH + 4, -HALF_HEIGHT + 4),
      createCone(HALF_WIDTH - 4, -HALF_HEIGHT + 4),
      createCone(-HALF_WIDTH + 4, HALF_HEIGHT - 4),
      createCone(HALF_WIDTH - 4, HALF_HEIGHT - 4)
    ];
    this.possession = 0; // 0 none, 1 player 1, 2 player 2
    this.winner = null;
    this.random = mulberry32(Math.floor(seed * 1e6));
    return this._observation();
  }

  step(actionIdx1, actionIdx2) {
    this.stepCount += 1;
    this.applyAction(0, actionIdx1);
    this.applyAction(1, actionIdx2);
    this.updatePhysics();
    const done = this.checkTerminal();
    return {
      observation: this._observation(),
      done,
      winner: this.winner,
      stepCount: this.stepCount
    };
  }

  applyAction(playerIdx, actionIdx) {
    const player = this.players[playerIdx];
    const action = ACTIONS[actionIdx] ?? "idle";
    const thrust = 18;
    let ax = 0;
    let ay = 0;
    switch (action) {
      case "up":
        ay = -thrust;
        break;
      case "down":
        ay = thrust;
        break;
      case "left":
        ax = -thrust;
        break;
      case "right":
        ax = thrust;
        break;
      case "upLeft":
        [ax, ay] = [-thrust, -thrust];
        break;
      case "upRight":
        [ax, ay] = [thrust, -thrust];
        break;
      case "downLeft":
        [ax, ay] = [-thrust, thrust];
        break;
      case "downRight":
        [ax, ay] = [thrust, thrust];
        break;
      case "kick":
        this.tryKick(playerIdx);
        break;
      default:
        break;
    }
    if (ax || ay) {
      const [nx, ny] = normalise(ax, ay);
      player.vx += nx * thrust * STEP_DT;
      player.vy += ny * thrust * STEP_DT;
      player.facing = Math.atan2(ny, nx);
    }
  }

  tryKick(playerIdx) {
    const player = this.players[playerIdx];
    const dx = this.ball.x - player.x;
    const dy = this.ball.y - player.y;
    const dist = vecLength(dx, dy);
    if (dist < PLAYER_RADIUS + BALL_RADIUS + 0.3) {
      const [nx, ny] = normalise(dx, dy);
      const impulse = 20;
      this.ball.vx = nx * impulse;
      this.ball.vy = ny * impulse;
      this.possession = playerIdx + 1;
    }
  }

  updatePhysics() {
    for (const player of this.players) {
      player.x += player.vx * STEP_DT;
      player.y += player.vy * STEP_DT;
      player.vx *= FRICTION;
      player.vy *= FRICTION;
      player.x = clamp(player.x, -HALF_WIDTH + PLAYER_RADIUS, HALF_WIDTH - PLAYER_RADIUS);
      player.y = clamp(player.y, -HALF_HEIGHT + PLAYER_RADIUS, HALF_HEIGHT - PLAYER_RADIUS);
    }

    this.ball.x += this.ball.vx * STEP_DT;
    this.ball.y += this.ball.vy * STEP_DT;
    this.ball.vx *= BALL_FRICTION;
    this.ball.vy *= BALL_FRICTION;

    // wall collisions for ball
    if (Math.abs(this.ball.x) > HALF_WIDTH - BALL_RADIUS) {
      this.ball.x = clamp(this.ball.x, -HALF_WIDTH + BALL_RADIUS, HALF_WIDTH - BALL_RADIUS);
      this.ball.vx *= -0.8;
    }
    if (Math.abs(this.ball.y) > HALF_HEIGHT - BALL_RADIUS) {
      this.ball.y = clamp(this.ball.y, -HALF_HEIGHT + BALL_RADIUS, HALF_HEIGHT - BALL_RADIUS);
      this.ball.vy *= -0.8;
    }

    this.resolvePlayerBall();
    this.resolvePlayers();
    this.resolveCones();
  }

  resolvePlayerBall() {
    for (let i = 0; i < this.players.length; i += 1) {
      const p = this.players[i];
      const dx = this.ball.x - p.x;
      const dy = this.ball.y - p.y;
      const dist = vecLength(dx, dy);
      const minDist = PLAYER_RADIUS + BALL_RADIUS;
      if (dist < minDist && dist > 0) {
        const overlap = minDist - dist;
        const [nx, ny] = [dx / dist, dy / dist];
        this.ball.x += nx * overlap;
        this.ball.y += ny * overlap;
        this.ball.vx += nx * overlap * 8;
        this.ball.vy += ny * overlap * 8;
        this.possession = i + 1;
      }
    }
  }

  resolvePlayers() {
    const [a, b] = this.players;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = vecLength(dx, dy);
    const minDist = PLAYER_RADIUS * 2;
    if (dist < minDist && dist > 0) {
      const overlap = minDist - dist;
      const [nx, ny] = [dx / dist, dy / dist];
      a.x -= nx * overlap * 0.5;
      a.y -= ny * overlap * 0.5;
      b.x += nx * overlap * 0.5;
      b.y += ny * overlap * 0.5;
      const impulse = overlap * 6;
      a.vx -= nx * impulse;
      a.vy -= ny * impulse;
      b.vx += nx * impulse;
      b.vy += ny * impulse;
    }
  }

  resolveCones() {
    for (const cone of this.cones) {
      cone.x += cone.vx * STEP_DT;
      cone.y += cone.vy * STEP_DT;
      cone.vx *= 0.9;
      cone.vy *= 0.9;
      cone.angle += cone.angularVelocity;
      cone.angularVelocity *= 0.92;
      if (Math.abs(cone.x) > HALF_WIDTH - CONE_RADIUS) {
        cone.x = clamp(cone.x, -HALF_WIDTH + CONE_RADIUS, HALF_WIDTH - CONE_RADIUS);
        cone.vx *= -0.5;
      }
      if (Math.abs(cone.y) > HALF_HEIGHT - CONE_RADIUS) {
        cone.y = clamp(cone.y, -HALF_HEIGHT + CONE_RADIUS, HALF_HEIGHT - CONE_RADIUS);
        cone.vy *= -0.5;
      }
      this.resolveBodyCollision(cone, this.ball, BALL_RADIUS, 0.4);
      for (const player of this.players) {
        this.resolveBodyCollision(cone, player, PLAYER_RADIUS, 0.6);
      }
    }
  }

  resolveBodyCollision(body, other, otherRadius, bounce) {
    const dx = other.x - body.x;
    const dy = other.y - body.y;
    const dist = vecLength(dx, dy);
    const minDist = CONE_RADIUS + otherRadius;
    if (dist < minDist && dist > 0) {
      const overlap = minDist - dist;
      const [nx, ny] = [dx / dist, dy / dist];
      other.x += nx * overlap;
      other.y += ny * overlap;
      other.vx += nx * overlap * 6;
      other.vy += ny * overlap * 6;
      body.vx -= nx * overlap * bounce;
      body.vy -= ny * overlap * bounce;
      body.angularVelocity += (Math.random() - 0.5) * 0.2;
    }
  }

  checkTerminal() {
    if (this.ball.x < -HALF_WIDTH && Math.abs(this.ball.y) < GOAL_HALF_WIDTH) {
      this.winner = 2; // right player scored on left goal
      return true;
    }
    if (this.ball.x > HALF_WIDTH && Math.abs(this.ball.y) < GOAL_HALF_WIDTH) {
      this.winner = 1;
      return true;
    }
    if (this.stepCount >= MAX_STEPS) {
      this.winner = 0;
      return true;
    }
    return false;
  }

  _observation() {
    const obs = [];
    for (const p of this.players) {
      obs.push(p.x / HALF_WIDTH, p.y / HALF_HEIGHT, p.vx, p.vy, Math.sin(p.facing));
    }
    obs.push(this.ball.x / HALF_WIDTH, this.ball.y / HALF_HEIGHT, this.ball.vx, this.ball.vy);
    for (const cone of this.cones) {
      obs.push(cone.x / HALF_WIDTH, cone.y / HALF_HEIGHT, cone.vx, cone.vy);
    }
    const possession = [0, 0, 0];
    possession[this.possession] = 1;
    obs.push(...possession);
    obs.push(1 - this.stepCount / MAX_STEPS);
    return obs;
  }
}

function createCone(x, y) {
  return {
    x,
    y,
    vx: 0,
    vy: 0,
    angle: 0,
    angularVelocity: 0
  };
}

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
