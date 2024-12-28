#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include <Eigen/Core>

#include <vector>

#ifdef TESTING
#include <gtest/gtest_prod.h>
#endif

namespace rnn::model {
enum class State {
  NORMAL,
  ERROR,
};
class InputLayer {
public:
  InputLayer(const Eigen::MatrixXd &inputs,
             const Eigen::MatrixXd &weightMatrix);

  static InputLayer CreateWithRandomWeights(const Eigen::MatrixXd &inputs,
                                            unsigned int hiddenSize);

  auto GetInput(unsigned int timeStepSize) -> Eigen::MatrixXd;

  auto WeightedSum(unsigned int timeStepSize) -> Eigen::MatrixXd;

  auto CalculateDeltasPerStep(unsigned int timeStep,
                              const Eigen::MatrixXd &deltaWeightedSum) -> void;

  auto UpdateWeightsAndBias(double learningRate) -> void;

  auto GetState() const -> State { return m_state; }

#ifdef TESTING
  FRIEND_TEST(TestCalculateDeltasPerStep, GivenValidInputs);
  FRIEND_TEST(TestUpdateWeightsAndBias, GivenValidInputs);
#endif
private:
  Eigen::MatrixXd m_inputs;
  Eigen::MatrixXd m_weightMatrix;
  Eigen::MatrixXd m_deltaWeights;
  State m_state;
};
} // namespace rnn::model
#endif
