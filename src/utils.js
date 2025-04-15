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

module.exports = {
  sigmoid,
  sigmoid_derivative,
  loss,
  dLoss
}


