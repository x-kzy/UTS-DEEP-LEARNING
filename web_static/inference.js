/* Inferensi MLP dan preprocessing di browser, tanpa backend Python.
   Seluruh parameter dibaca dari artefak hasil Section 13 notebook. */
(function (root) {
  'use strict';
  function assert(cond, message) { if (!cond) throw new Error(message); }
  function isBlank(value) { return value === null || value === undefined || value === ''; }

  function transformItem(raw, prep) {
    const item = Object.assign({}, raw);
    const numeric = prep.numerical_features;
    const cats = prep.categorical_features;
    assert(prep.numerical_means.length === numeric.length, 'Dimensi mean numerik tidak cocok.');
    assert(prep.numerical_scales.length === numeric.length, 'Dimensi skala numerik tidak cocok.');
    assert(prep.categories.length === cats.length, 'Dimensi encoder kategorikal tidak cocok.');
    for (const col of cats) {
      item[col] = String(isBlank(item[col]) ? '' : item[col]);
    }
    for (const col of numeric) {
      if (col === 'ppi') continue;
      if (isBlank(item[col])) {
        if (Object.prototype.hasOwnProperty.call(prep.imputation.global_medians, col)) {
          const group = item[prep.imputation.group_col];
          const candidate = prep.imputation.group_medians[col][group];
          item[col] = candidate === undefined ? prep.imputation.global_medians[col] : candidate;
        } else {
          item[col] = prep.numerical_defaults[col];
        }
      } else {
        item[col] = Number(item[col]);
      }
      assert(Number.isFinite(item[col]), 'Fitur numerik tidak valid: ' + col);
    }
    if (numeric.includes('ppi')) {
      assert(item.display_size > 0, 'Ukuran layar harus lebih dari nol.');
      item.ppi = Math.hypot(item.resolution_height, item.resolution_width) / item.display_size;
    }
    let vector = [];
    for (let i = 0; i < numeric.length; i++) {
      let value = Number(item[numeric[i]]);
      assert(Number.isFinite(value), 'Fitur numerik tidak valid: ' + numeric[i]);
      vector.push(Math.fround((value - prep.numerical_means[i]) / prep.numerical_scales[i]));
    }
    for (let i = 0; i < cats.length; i++) {
      for (const category of prep.categories[i]) {
        vector.push(item[cats[i]] === String(category) ? 1 : 0);
      }
    }
    assert(vector.length === prep.input_tensor_dim, 'Input model tidak sesuai dimensi preprocessing.');
    return { vector: vector, completed: item };
  }

  function runLayers(features, net) {
    let x = features.slice();
    for (const layer of net.layers) {
      if (layer.type === 'Dense') {
        assert(x.length === layer.input_dim, 'Dimensi Dense tidak sesuai.');
        const y = new Array(layer.output_dim);
        for (let j = 0; j < layer.output_dim; j++) {
          let sum = layer.bias[j];
          for (let i = 0; i < x.length; i++) sum += x[i] * layer.kernel[i][j];
          y[j] = sum;
        }
        x = y;
        if (layer.activation === 'relu') x = x.map(v => Math.max(0, v));
        else if (layer.activation === 'sigmoid') x = x.map(v => 1 / (1 + Math.exp(-v)));
        else if (layer.activation === 'tanh') x = x.map(v => Math.tanh(v));
        else if (layer.activation === 'softmax') x = softmax(x);
        else assert(!layer.activation || layer.activation === 'linear', 'Aktivasi Dense tidak didukung: ' + layer.activation);
      } else if (layer.type === 'BatchNormalization') {
        assert(x.length === layer.mean.length, 'Dimensi BatchNormalization tidak cocok.');
        x = x.map((v, i) => (v - layer.mean[i]) * layer.gamma[i] /
          Math.sqrt(layer.variance[i] + layer.epsilon) + layer.beta[i]);
      } else if (layer.type === 'Activation') {
        if (layer.activation === 'relu') x = x.map(v => Math.max(0, v));
        else if (layer.activation === 'softmax') x = softmax(x);
        else if (layer.activation === 'tanh') x = x.map(v => Math.tanh(v));
        else if (layer.activation === 'linear') {}
        else throw new Error('Aktivasi tidak didukung: ' + layer.activation);
      } else if (layer.type === 'Dropout') {
        // Dropout nonaktif pada mode inference; tidak mengubah nilai aktivasi.
      } else {
        throw new Error('Jenis layer tidak didukung: ' + layer.type);
      }
    }
    assert(x.every(Number.isFinite), 'Probabilitas prediksi tidak valid.');
    return x;
  }

  function softmax(z) {
    const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / sum);
  }

  function predict(raw, prep, net, labels, schema) {
    const t = transformItem(raw, prep);
    const probabilities = runLayers(t.vector, net);
    let classIndex = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[classIndex];
    const outside = [];
    for (const col of prep.numerical_features) {
      if (col === 'ppi') continue;
      const limits = schema.training_ranges && schema.training_ranges[col];
      if (limits && (t.completed[col] < limits.min || t.completed[col] > limits.max)) outside.push(col);
    }
    return {
      classIndex,
      probabilities,
      confidence,
      classInfo: labels.classes[String(classIndex)],
      requiresReview: confidence < Number(schema.confidence_threshold_tau),
      outsideTrainingRange: outside,
      featureVector: t.vector
    };
  }
  root.LaptopInference = { transformItem, runLayers, predict };
})(typeof globalThis !== 'undefined' ? globalThis : this);
