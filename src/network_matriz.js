const math = require('mathjs');
const { sigmoid, sigmoid_derivative } = require('./utils');

class Network {
  constructor(sizes) {
    this.sizes = sizes
    this.numLayers = sizes.length
    this.weights = []
    this.biases = []
    this.a = []
    this.deltas = [null]
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
    for (let l = L - 1; l > 0; l--) {
      const W_next_T = math.transpose(this.weights[l]);
      const delta_next = this.deltas[l + 1]; 
      const z = this.zs[l - 1];
      const ad = z.map(([val]) => [sigmoid_derivative(val)]);

      const delta = math.dotMultiply(
        math.multiply(W_next_T, delta_next),
        ad
      );
      this.deltas[l] = delta;
    }
  }

  updateWeights(learningRate) {
    const L = this.numLayers - 1;

    for (let l = L - 1; l >= 1; l--) {
      const delta = this.deltas[l];
      const a_prev = this.a[l - 1];

      const dW = math.multiply(delta, math.transpose(a_prev));
      const db = delta;

      this.weights[l - 1] = math.subtract(this.weights[l - 1], math.multiply(learningRate, dW));
      this.biases[l - 1] = math.subtract(this.biases[l - 1], math.multiply(learningRate, db));
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

