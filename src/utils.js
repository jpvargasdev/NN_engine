function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoid_derivative(x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

function loss(yHat, y) {
  return (yHat - y) ** 2;
}

function dLoss(yHat, y) {
  return 2 * (yHat - y);
}

function crossEntropyError(yHat, y) {
  const epsilon = 1e-12;
  const a = Math.min(Math.max(yHat, epsilon), 1 - epsilon); 
  return -(y * Math.log(a) + (1 - y) * Math.log(1 - a));
}

function dCrossEntropy(yHat, y) {
  const epsilon = 1e-12;
  const a = Math.min(Math.max(yHat, epsilon), 1 - epsilon);
  return (a - y) / (a * (1 - a));
}

function relu(x) {
  return Math.max(0, x);
}

function dRelu(x) {
  return x > 0 ? 1 : 0;
}

module.exports = {
  sigmoid,
  sigmoid_derivative,
  loss,
  dLoss,
  crossEntropyError,
  dCrossEntropy,
  relu,
  dRelu
}


