export const FIELD_WIDTH = 60;
export const FIELD_HEIGHT = 36;
export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 480;
export const TICK_RATE = 60;
export const MAX_STEPS = TICK_RATE * 30; // 30 seconds
export const BALL_RADIUS = 1.2;
export const PLAYER_RADIUS = 1.5;
export const CONE_RADIUS = 0.8;
export const FRICTION = 0.96;
export const BALL_FRICTION = 0.985;
export const STEP_DT = 1 / TICK_RATE;
export const GOAL_HALF_WIDTH = 5;
export const ACTIONS = [
  "idle",
  "up",
  "down",
  "left",
  "right",
  "upLeft",
  "upRight",
  "downLeft",
  "downRight",
  "kick"
];
export const ATTRIBUTES = [
  { name: "speed", base: 0.5 },
  { name: "strength", base: 0.5 },
  { name: "stamina", base: 0.5 },
  { name: "agility", base: 0.5 },
  { name: "vision", base: 0.5 },
  { name: "accuracy", base: 0.5 },
  { name: "control", base: 0.5 }
];
export const INITIAL_ATTRIBUTE_SCORE = 5; // out of 10
export const REWARD_WIN = 500;
export const REWARD_LOSS = -500;
export const REWARD_DRAW = -50;
export const OBSERVATION_SIZE =
  5 * 2 + // players pos/vel/facing
  4 + // ball pos/vel
  4 * 4 + // cones (4 states * 4 values)
  3 + // possession one-hot
  1; // time remaining
export const EPSILON_START = 0.2;
export const EPSILON_END = 0.05;
export const EPSILON_DECAY_STEPS = 10000;
export const GAMMA = 0.99;
export const LEARNING_RATE = 0.00025;
export const TARGET_SYNC_INTERVAL = 2000;
export const LOCAL_BATCH_SIZE = 32;
export const LOCAL_UPDATE_STEPS = 2;
export const REFRESH_INTERVAL_MS = 45000;
export const TRAIN_EPISODES_PER_LOOP = 8;
export const MAX_REPLAY_SIZE = 50000;
