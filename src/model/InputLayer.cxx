#include "InputLayer.h"

#include <random>

auto GenerateRandomMatrix(unsigned int rows,
                          unsigned int cols) -> Eigen::MatrixXd {
  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<> distr(0, 1);
  Eigen::MatrixXd matrix(rows, cols);
  std::generate(matrix.data(), matrix.data() + matrix.size(),
                [&]() { return distr(eng); });
  return matrix;
}

namespace rnn::model {
InputLayer::InputLayer(const Eigen::MatrixXd &inputs,
                       const Eigen::MatrixXd &weightMatrix)
    : m_inputs(inputs), m_weightMatrix(weightMatrix),
      m_deltaWeights(Eigen::MatrixXd::Zero(weightMatrix.rows(), inputs.cols())),
      m_state(State::NORMAL) {}

InputLayer InputLayer::CreateWithRandomWeights(const Eigen::MatrixXd &inputs,
                                               unsigned int hiddenSize) {
  auto weights = GenerateRandomMatrix(hiddenSize, inputs.size());
  return InputLayer(inputs, weights);
}

auto InputLayer::GetInput(unsigned int timeStepSize) -> Eigen::MatrixXd {
  if (timeStepSize >= m_inputs.rows()) {
    m_state = State::ERROR;
    return Eigen::MatrixXd();
  }
  return m_inputs.row(timeStepSize);
}

auto InputLayer::WeightedSum(unsigned int timeStepSize) -> Eigen::MatrixXd {
  if (timeStepSize >= m_inputs.rows()) {
    m_state = State::ERROR;
    return Eigen::MatrixXd();
  }
  return m_weightMatrix * m_inputs.row(timeStepSize).transpose();
}

auto InputLayer::CalculateDeltasPerStep(
    unsigned int timeStep, const Eigen::MatrixXd &deltaWeightedSum) -> void {
  auto input = GetInput(timeStep);
  if (input.size() == 0 || deltaWeightedSum.cols() != input.rows() ||
      deltaWeightedSum.rows() != m_deltaWeights.rows()) {
    m_state = State::ERROR;
    return;
  }
  m_deltaWeights += deltaWeightedSum * input;
}

auto InputLayer::UpdateWeightsAndBias(double learningRate) -> void {
  m_weightMatrix -= learningRate * m_deltaWeights;
}

} // namespace rnn::model
