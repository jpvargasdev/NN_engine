const { Network } = require('./network');
const math = require('mathjs');
const { generateGrid } = require('./generate_grid');

const xorData = [
  { input: [[0], [0]], output: [[0]] },
  { input: [[0], [1]], output: [[1]] },
  { input: [[1], [0]], output: [[1]] },
  { input: [[1], [1]], output: [[0]] },
];

const network = new Network([2, 10, 10, 1]); 
const maxEpochs = 40000;
const learningRate = 3;
const targetLoss = 0.01;
const lossHistory = [];

for (let epoch = 0; epoch < maxEpochs; epoch++) {
  let totalLoss = 0;

  xorData.forEach(({ input, output }) => {
    const prediction = network.train(input, output, learningRate);
    const diff = math.subtract(prediction, output);
    const squared = math.map(diff, val => val * val);
    const loss = math.sum(squared);
    totalLoss += loss;
  });

  const avgLoss = totalLoss / xorData.length;
  lossHistory.push(avgLoss);

  if (epoch % 500 === 0) {
    console.log(`Epoch ${epoch} → Loss: ${avgLoss.toFixed(6)}`);
  }

  if (avgLoss < targetLoss) {
    console.log(`Early stopping at epoch ${epoch} → Loss: ${avgLoss.toFixed(6)}`);
    break;
  }
}

console.log('\nTesting XOR:');
xorData.forEach(({ input }) => {
  const output = network.forward(input);
  console.log(`Input: ${input.flat()} → Pred: ${output[0][0].toFixed(4)}`);
});

generateGrid(network);
