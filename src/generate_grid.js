const fs = require('fs');

function generateGrid(network) {
  const resolution = 100;
  const range = [...Array(resolution).keys()].map(i => i / (resolution - 1));
  const predictions = [];

  range.forEach(x => {
    range.forEach(y => {
      const output = network.forward([[x], [y]])[0][0]; 
      predictions.push({ x, y, z: output });
    });
  });

  const xorPoints = [
    { x: 0, y: 0, expected: 0 },
    { x: 0, y: 1, expected: 1 },
    { x: 1, y: 0, expected: 1 },
    { x: 1, y: 1, expected: 0 },
  ];

  xorPoints.forEach(({ x, y, expected }) => {
    const z = network.forward([[x], [y]])[0][0];
    predictions.push({ x, y, z, expected });
  });

  fs.writeFileSync('predictions.json', JSON.stringify(predictions, null, 2), 'utf-8');
  console.log('✅ predictions.json done');
}

module.exports = {
  generateGrid
};
