#include "model/InputLayer.h"

int main() {
  auto inputs = Eigen::MatrixXd::Zero(10, 10);
  rnn::model::InputLayer layer =
      rnn::model::InputLayer::CreateWithRandomWeights(inputs, 1);
  layer.UpdateWeightsAndBias(1.0);
  return 0;
}
