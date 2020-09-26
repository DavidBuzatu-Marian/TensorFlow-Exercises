/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */

getData = async () => {
  const carsDataResponse = await fetch(
    'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
  );
  const carsData = await carsDataResponse.json();
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);

  return cleaned;
};

run = async () => {
  const data = await getData();
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLavel: 'MPG',
      height: 300,
    }
  );

  const model = createModel();
  tfvis.show.modelSummary({ name: 'Model Summary' }, model);

  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  await trainModel(model, inputs, labels);
  console.log('Done training!');

  testModel(model, data, tensorData);
};

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
convertToTensor = (data) => {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
};

trainModel = async (model, inputs, labels) => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 64;
  const epochs = 100;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    ),
  });
};

createModel = () => {
  const model = tf.sequential();

  // single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 10, useBias: true }));

  model.add(tf.layers.dense({ units: 30, activation: 'sigmoid' }));
  model.add(tf.layers.dense({ units: 30, activation: 'sigmoid' }));
  model.add(tf.layers.dense({ units: 10, activation: 'sigmoid' }));
  // output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
};

testModel = (model, inputData, normalizationData) => {
  const { inputMax, inputMin, labelMax, labelMin } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    {
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted'],
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    }
  );
};

document.addEventListener('DOMContentLoaded', run);
