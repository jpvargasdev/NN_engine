const { Network: NetworkPOO} = require('./network')
const { Network: NetworkMatriz } = require('./network_matriz')
const Plotly = require('plotly.js')

const nn = [2, 10, 10, 1]; // Now we have a hidden layer of 3

const networkPoo = new NetworkPOO(nn);
const networkMatriz = new NetworkMatriz(nn);

// for (let i = 0; i < 10; i++) {
//   const x = [1, 1];
//   const y = [0.25];
//   const output = networkPoo.train(x, y, 3);
//   console.log('Output', output);
// }

for (let i = 0; i < 10; i++) {
  const x = [[1], [1]];
  const y = [[0.25]];
  const output = networkMatriz.train(x, y, 3);
  console.log('Output', output);
}
