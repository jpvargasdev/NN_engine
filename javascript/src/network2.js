const math = require('mathjs');
const { sigmoid, sigmoid_derivative, relu, dRelu } = require('./utils');

class Network2 {
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
      let a = 0;
      if (index === this.weights.length - 1) {
        a = z.map(([val]) => [sigmoid(val)]);
      } else {
        a = z.map(([val]) => [relu[val]]);
      }
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

    // 1.1 - calculate delta using simplified gradient 
    const delta_output = math.subtract(yHat, y);

    this.deltas[L] = delta_output;

    // 2 - propagate error backwards
    for (let layerIndex = L - 1; layerIndex > 0; layerIndex--) {
      const W_next_T = math.transpose(this.weights[layerIndex]);
      const delta_next = this.deltas[layerIndex + 1]; 
      const z = this.zs[layerIndex - 1];

      let ad = 0;
      if (layerIndex === this.weights.length - 1) {
        ad = z.map(([val]) => [sigmoid_derivative(val)]);
      } else {
        ad = z.map(([val]) => [dRelu(val)]);
      }

      const delta = math.dotMultiply(
        math.multiply(W_next_T, delta_next),
        ad
      );
      this.deltas[layerIndex] = delta;
    }
  }

  updateWeights(learningRate, lambda = 0.5) {
    const L = this.numLayers - 1;
      for (let layerIndex = L - 1; layerIndex >= 1; layerIndex--) {
      const delta = this.deltas[layerIndex];
      const a_prev = this.a[layerIndex - 1];

      const dW = math.multiply(delta, math.transpose(a_prev));
      const db = delta;

      // L2 regularization
      const regTerm = math.multiply(lambda, this.weights[layerIndex - 1]);
      const dw_with_reg = math.add(dW, regTerm);

      this.weights[layerIndex - 1] = math.subtract(
        this.weights[layerIndex - 1],
        math.multiply(learningRate, dw_with_reg)
      );
      this.biases[layerIndex - 1] = math.subtract(
        this.biases[layerIndex - 1],
        math.multiply(learningRate, db)
      );
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
  Network2
}

