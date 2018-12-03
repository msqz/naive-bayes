#include "classifier.h"
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <vector>

/**
 * Initializes GNB
 */
GNB::GNB() {}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels) {
  /*
          Trains the classifier with N data points and labels.

          INPUTS
          data - array of N observations
            - Each observation is a tuple with 4 values: s, d,
              s_dot and d_dot.
            - Example : [
                          [3.5, 0.1, 5.9, -0.02],
                          [8.0, -0.3, 3.0, 2.2],
                          ...
                  ]

          labels - array of N labels
            - Each label is one of "left", "keep", or "right".
  */
  uint n_labels = possible_labels.size();
  means.reserve(n_labels);
  stds.reserve(n_labels);
  uint n_features = data[0].size();
  uint n_samples = data.size();
  std::vector<uint> cardinalities(n_labels, 0u);
  std::map<string, uint> labels_to_n;

  for (uint i_label = 0; i_label < n_labels; i_label++) {
    labels_to_n[possible_labels[i_label]] = i_label;
    std::vector<double> features_init(n_features, 0.0);
    means.push_back(features_init);
    stds.push_back(features_init);
  }

  // calculating means
  for (uint i_sample = 0; i_sample < n_samples; i_sample++) {
    uint idx_label = labels_to_n[labels[i_sample]];
    for (uint i_feature = 0; i_feature < n_features; i_feature++) {
      means[idx_label][i_feature] += data[i_sample][i_feature];
    }
    cardinalities[idx_label] += 1;
  }

  for (uint i_label = 0; i_label < n_labels; i_label++) {
    for (uint i_feature = 0; i_feature < n_features; i_feature++) {
      means[i_label][i_feature] /= cardinalities[i_label];
    }
  }
  // endof calculating means
  for (uint i_sample = 0; i_sample < n_samples; i_sample++) {
    uint idx_label = labels_to_n[labels[i_sample]];
    for (uint i_feature = 0; i_feature < n_features; i_feature++) {
      stds[idx_label][i_feature] += pow(data[i_sample][i_feature] - means[idx_label][i_feature], 2);
    }
  }
  for (uint i_label = 0; i_label < n_labels; i_label++) {
    for (uint i_feature = 0; i_feature < n_features; i_feature++) {
      stds[i_label][i_feature] = sqrt(stds[i_label][i_feature] / cardinalities[i_label]);
    }
  }
  // calculating stds
}

string GNB::predict(vector<double> sample) {
  /*
          Once trained, this method is called and expected to return
          a predicted behavior for the given observation.

          INPUTS

          observation - a 4 tuple with s, d, s_dot, d_dot.
            - Example: [3.5, 0.1, 8.5, -0.2]

          OUTPUT

          A label representing the best guess of the classifier. Can
          be one of "left", "keep" or "right".
          """
          # TODO - complete this
  */
  uint n_labels = possible_labels.size();
  uint n_features = sample.size();
  vector<double> probabilities(n_labels, 1);

  for (uint i_label = 0; i_label < n_labels; i_label++) {
    for (uint i_feature = 0; i_feature < n_features; i_feature++) {
      probabilities[i_label] *= gaussian(sample[i_feature],
                                         means[i_label][i_feature],
                                         stds[i_label][i_feature]);
    }
  }

  auto max_probability = std::max_element(probabilities.begin(), probabilities.end());
  // std::cout << *max_probability << "\n";
  int idx_label = std::distance(probabilities.begin(), max_probability);

  return this->possible_labels[idx_label];
}

double GNB::gaussian(double x, double mean, double std) {
  // cout << x << ", " << mean << ", " << std << "\n";
  double exponent = exp(-(pow(x - mean, 2)) / (2 * pow(std, 2)));
  double denominator = 2 * M_PI * pow(std, 2);
  return exponent / denominator;
}