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
  means.reserve(possible_labels.size());
  stds.reserve(possible_labels.size());

  for (int i_label = 0; i_label < possible_labels.size(); i_label++) {
    means[i_label].reserve(data[0].size());
    stds[i_label].reserve(data[0].size());

    for (int i_feature = 0; i_feature < data[0].size(); i_feature++) {
      for (int i_data = 0; i_data < data.size(); i_data++) {
        means[i_label][i_feature] += data[i_data][i_feature];
      }
      means[i_label][i_feature] /= data.size();

      double sqr_sum = 0.0;
      for (int i_data = 0; i_data < data.size(); i_data++) {
        sqr_sum += pow(data[i_data][i_feature] - means[i_label][i_feature], 2);
      }
      stds[i_label][i_feature] = sqrt(sqr_sum / data.size());
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
  vector<double> probabilities(3, 1);

  for (int i_label = 0; i_label < possible_labels.size(); i_label++) {
    for (int i_feature = 0; i_feature < sample.size(); i_feature++) {
      probabilities[i_label] *=
          gaussian(sample[i_feature], means[i_label][i_feature],
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