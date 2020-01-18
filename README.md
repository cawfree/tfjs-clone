# tfjs-clone
Creates a deep clone of an existing @tensorflow/tfjs model.

<a href="#badge">
  <img alt="code style: prettier" src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square">
</a>

## 🚀 Getting Started

Using [`npm`]():

```bash
npm install --save tfjs-clone
```

Using [`yarn`]():

```bash
yarn add tfjs-clone
```

## ✍️ Usage

```javascript
import * as tf from '@tensorflow/tfjs';
import { model as clone } from 'tfjs-clone';

const m = tf.sequential();

m.add(tf.layers.dense({ inputShape: [1], units: 16, activation: 'relu' }));
m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
m.compile({ optimizer: tf.train.rmsprop(1e-2), loss: 'binaryCrossentropy' });
  
const m2 = await clone(m);
```
## ✌️ License
[MIT](https://opensource.org/licenses/MIT)
