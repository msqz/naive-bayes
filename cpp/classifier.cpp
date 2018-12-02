#include "classifier.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <iterator>
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
  int n_labels = possible_labels.size();
  int n_features = data[0].size();
  int n_samples = data.size();

  means.insert(means.begin(), n_labels, std::vector<double>(n_samples, 0.0));
  stds.insert(stds.begin(), n_labels, std::vector<double>(n_samples, 0.0));

  for (int i_label = 0; i_label < n_labels; i_label++) {
    for (int i_feature = 0; i_feature < n_features; i_feature++) {
      for (int i_sample = 0; i_sample < n_samples; i_sample++) {
        means[i_label][i_feature] += data[i_sample][i_feature];
      }
      means[i_label][i_feature] /= n_samples;

      double sqr_sum = 0.0;
      for (int i_sample = 0; i_sample < n_samples; i_sample++) {
        sqr_sum += pow(data[i_sample][i_feature] - means[i_label][i_feature], 2);
      }

      stds[i_label][i_feature] = sqrt(sqr_sum / n_samples);
    }
  }
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
  int n_labels = possible_labels.size();
  int n_features = sample.size();
  vector<double> probabilities(3, 1);

  for (int i_label = 0; i_label < n_labels; i_label++) {
    for (int i_feature = 0; i_feature < n_features; i_feature++) {
      probabilities[i_label] *= gaussian(sample[i_feature],
                                         means[i_label][i_feature],
                                         stds[i_label][i_feature]);
    }
  }

  auto max_probability = std::max(probabilities.begin(), probabilities.end());
  // std::cout << *max_probability << "\n";
  int idx_label = std::distance(probabilities.begin(), max_probability);

  return this->possible_labels[idx_label];
}

double GNB::gaussian(double x, double mean, double std) {
  // cout << x << ", " << mean << ", " << std << "\n";
  double exponent = exp(-(pow(x - mean, 2) / (2 * pow(std, 2))));
  double denominator = 2 * M_PI * pow(std, 2);
  return exponent / denominator;
}