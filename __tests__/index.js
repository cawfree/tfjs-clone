import '@babel/polyfill';
import * as tf from '@tensorflow/tfjs';

import { model as clone } from '../src';

it('should safely clone a model', async () => {
  const m = tf.sequential();
  
  m.add(tf.layers.dense({ inputShape: [1], units: 2, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  
  const m2 = await clone(m);

  expect(JSON.stringify(m))
    .toEqual(JSON.stringify(m2));

  const r1 = m.predict(tf.ones([2]));
  const r2 = m2.predict(tf.ones([2]));

  expect(JSON.stringify(r1.dataSync()))
    .toEqual(JSON.stringify(r2.dataSync()));

  m.compile({ optimizer: tf.train.rmsprop(1e-2), loss: 'binaryCrossentropy' });
  m2.compile({ optimizer: tf.train.rmsprop(1e-2), loss: 'binaryCrossentropy' });

  await m2.fit(
    tf.tensor2d([[1], [0]]),
    tf.tensor2d([[0], [1]]),
    {
      epochs: 1,
      batchSize: 1,
    },
  );

  const r3 = m.predict(tf.ones([2]));
  const r4 = m2.predict(tf.ones([2]));

  expect(JSON.stringify(r3.dataSync()))
    .not
    .toEqual(JSON.stringify(r4.dataSync()));

});
