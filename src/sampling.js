const FIRST_WEIGHTS = [
  { who: "gregory", weight: 1.0 },
  { who: "fred", weight: 1.5 }
];

export function pickPlayers(rng = Math.random) {
  const total = FIRST_WEIGHTS.reduce((acc, { weight }) => acc + weight, 0);
  const r = rng() * total;
  let sum = 0;
  let first = FIRST_WEIGHTS[0].who;
  for (const entry of FIRST_WEIGHTS) {
    sum += entry.weight;
    if (r <= sum) {
      first = entry.who;
      break;
    }
  }
  let second;
  if (first === "gregory") {
    second = "fred";
  } else {
    const options = ["gregory", "fred"];
    second = options[Math.floor(rng() * options.length)];
  }
  return { p1Who: first, p2Who: second };
}
