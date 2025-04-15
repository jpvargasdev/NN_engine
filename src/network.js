const utils = require("./utils");
const { sigmoid, loss, dLoss } = utils;

class Network {
  constructor(layers, inputs) {
    this.mLayers = [];
    for (let i = 0; i < layers.length; i++) {
      const numInputs = layers[i - 1] || 0;
      const numNeuros = layers[i];
      if (i === 0) {
        // first layer receives the inputs
        this.mLayers.push(new Layer(numNeuros, inputs));
      } else {
        this.mLayers.push(new Layer(numNeuros, numInputs));
      }
    }
  }

  forward(inputs) {
    let activations = inputs;
    this.mLayers.forEach((layer) => {
      activations = layer.forward(activations);
    })
    return activations
  }

  train(inputs, yTrue, learningRate) {
    this.forward(inputs);
    for (let i = this.mLayers.length - 1; i >= 0; i--) {
      const layer = this.mLayers[i];

      if (i === this.mLayers.length - 1) {
        // Output layer
        layer.neurons.forEach((neuron, j) => {
          neuron.backward(yTrue[j], learningRate, true);
        });
      } else {
        const nextLayer = this.mLayers[i + 1];
        layer.neurons.forEach((neuron) => {
          neuron.backward(nextLayer, learningRate, false);
        });
      }
    }  

    return this.mLayers[this.mLayers.length - 1].neurons.map((neuron) => {
      return neuron.a;
    })
  }
}

class Layer {
  constructor(neurons, inputs) {
    this.neurons = [];
    this.numInputs = inputs;
    this.inputs = inputs;
    for (let i = 0; i < neurons; i++) {
      const neuron = new Neuron(inputs);
      neuron.index = i; // 📌 Needed for hidden layer delta calculation
      this.neurons.push(neuron);
    }
  }

  forward(inputs) {
    this.inputs = inputs;
    return this.neurons.map((neuron) => {
      return neuron.activate(this.inputs);
    })
  } 
}

class Neuron {
  constructor(numInputs) {
    this.z = 0;
    this.yHat = 0;
    this.delta = 0;
    this.weights = Array.from({ length: numInputs }, () => Math.random() * 2 - 1);
    this.bias = Math.random() * 2 - 1;
  }

  activate(inputs) {
    this.inputs = inputs;
    this.z = 0;
    this.a = 0;
    if (this.weights.length === inputs.length) {
      this.weights.forEach((weight, index) => {
        this.z += weight * inputs[index];
      })
      this.z += this.bias;
      this.a = sigmoid(this.z);
    }

    return this.a
  }
  
  backward(targetOrNextLayer, learningRate, isOutputLayer = false) {
    if (isOutputLayer) {
      const dL_da = dLoss(this.a, targetOrNextLayer);
      const dA_dz = this.a * (1 - this.a);
      this.delta = dL_da * dA_dz;
    } else {
      let sum = 0;
      // O(n)
      for (let j = 0; j < targetOrNextLayer.neurons.length; j++) {
        const nextLayerNeuron = targetOrNextLayer.neurons[j];
        sum += nextLayerNeuron.weights[this.index] * nextLayerNeuron.delta;
      }

      const dA_dZ = this.a * (1 - this.a);
      this.delta = sum * dA_dZ;
    }

    // O(n)
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] -= learningRate * this.delta * this.inputs[i];
    }

    // Update bias: b = b - α * ∂L/∂z
    this.bias -= learningRate * this.delta;
  }
}

module.exports = {
  Network
}


