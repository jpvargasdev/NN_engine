const math = require('mathjs');
const { sigmoid, sigmoid_derivative } = require('./utils');

class Network {
  constructor(sizes) {
    this.sizes = sizes
    this.numLayers = sizes.length
    this.weights = []
    this.biases = []
    this.a = []
    this.deltas = new Array(this.numLayers).fill(null)
    this.zs = []

    this.sizes.slice(0, -1).forEach((size, index) => {
      const rows = this.sizes[index + 1];
      const cols = size;

      const W = math.random([rows, cols], -1, 1);
      const b = math.random([rows, 1], -1, 1);
     
      this.weights.push(W);
      this.biases.push(b);
    })
  } 

  forward(inputs) {
    this.a = [inputs];
    this.zs = []
    let nextActivation = inputs

    this.weights.forEach((weight, index) => {
      const z = math.add(math.multiply(weight, nextActivation), this.biases[index]);
      const a = z.map(([val]) => [sigmoid(val)]);
      this.zs.push(z);
      this.a.push(a);
      nextActivation = a
    })

    return this.a[this.a.length - 1];
  }

  backward(y) {
    // 1 - calculate error in output layer
    const L = this.numLayers - 1;
    const yHat = this.a[this.a.length - 1];
    const delta_output = math.dotMultiply(
      math.subtract(yHat, y),
      yHat.map(([val]) => [val * (1 - val)])
    );

    this.deltas[L] = delta_output;

    // 2 - propagate error backwards
    for (let layerIndex = L - 1; layerIndex > 0; layerIndex--) {
      const W_next_T = math.transpose(this.weights[layerIndex]);
      const delta_next = this.deltas[layerIndex + 1]; 
      const z = this.zs[layerIndex - 1];
      const ad = z.map(([val]) => [sigmoid_derivative(val)]);

      const delta = math.dotMultiply(
        math.multiply(W_next_T, delta_next),
        ad
      );
      this.deltas[layerIndex] = delta;
    }
  }

  updateWeights(learningRate) {
    const L = this.numLayers - 1;

    for (let layerIndex = L - 1; layerIndex >= 1; layerIndex--) {
      const delta = this.deltas[layerIndex];
      const a_prev = this.a[layerIndex - 1];

      const dW = math.multiply(delta, math.transpose(a_prev));
      const db = delta;

      this.weights[layerIndex - 1] = math.subtract(this.weights[layerIndex - 1], math.multiply(learningRate, dW));
      this.biases[layerIndex - 1] = math.subtract(this.biases[layerIndex - 1], math.multiply(learningRate, db));
    }
  }

  train(inputs, y, learningRate) {
    this.forward(inputs);
    this.backward(y);
    this.updateWeights(learningRate);
    return this.a[this.a.length - 1];
  }
}
module.exports = {
  Network
}

