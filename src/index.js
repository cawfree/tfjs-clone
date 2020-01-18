import * as tf from "@tensorflow/tfjs";

export const model = model =>
  new Promise(resolve => model.save({ save: resolve })).then(modelData =>
    tf.loadLayersModel({ load: () => modelData })
  );

export const tensor = tensor => tensor.clone();
