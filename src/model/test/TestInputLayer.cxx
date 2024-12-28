#define TESTING
#include "InputLayer.h"

#include <gtest/gtest.h>

namespace rnn::model {
struct TestParams {
  Eigen::MatrixXd inputMatrix;
  int index;
  Eigen::MatrixXd expectedOutput;
  State expectedState;
};

class InputLayerTest_GetInput : public ::testing::TestWithParam<TestParams> {};

TEST_P(InputLayerTest_GetInput, GetInput) {
  const TestParams &params = GetParam();
  Eigen::MatrixXd weights;
  InputLayer layer(params.inputMatrix, weights);

  Eigen::MatrixXd result = layer.GetInput(params.index);

  ASSERT_TRUE(result.isApprox(params.expectedOutput));
  ASSERT_EQ(params.expectedState, layer.GetState());
}

INSTANTIATE_TEST_SUITE_P(
    InputLayerTest_GetInput, InputLayerTest_GetInput,
    ::testing::Values(
        TestParams{Eigen::MatrixXd::Random(3, 3), 3, Eigen::MatrixXd(0, 0),
                   State::ERROR}, // Index out of bounds
        TestParams{
            (Eigen::MatrixXd(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished(), 1,
            (Eigen::MatrixXd(1, 3) << 4, 5, 6).finished(), State::NORMAL}
        // Valid index
        ));

class InputLayerTest_WeightedSum : public ::testing::TestWithParam<TestParams> {
};

TEST_P(InputLayerTest_WeightedSum, WeightedSum) {
  const TestParams &params = GetParam();
  auto weights = (Eigen::MatrixXd(2, 3) << 1, 2, 3, 4, 5, 6).finished();
  InputLayer layer(params.inputMatrix, weights);

  Eigen::MatrixXd result = layer.WeightedSum(params.index);

  ASSERT_TRUE(result.isApprox(params.expectedOutput));
  ASSERT_EQ(params.expectedState, layer.GetState());
}

INSTANTIATE_TEST_SUITE_P(
    InputLayerTest_WeightedSum, InputLayerTest_WeightedSum,
    ::testing::Values(
        TestParams{Eigen::MatrixXd::Random(3, 3), 3, Eigen::MatrixXd(0, 0),
                   State::ERROR}, // Index out of bounds
        TestParams{
            (Eigen::MatrixXd(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished(), 1,
            (Eigen::MatrixXd(2, 1) << 32, 77).finished(), State::NORMAL}));

struct TestParamsForCalcDeltas {
  Eigen::MatrixXd inputs;
  int index;
  Eigen::MatrixXd deltaWeightedSum;
  State expectedState;
};

class InputLayerTest_CalculateDeltasPerStep
    : public ::testing::TestWithParam<TestParamsForCalcDeltas> {};

TEST_P(InputLayerTest_CalculateDeltasPerStep, CalculateDeltasPerStep) {
  const auto &params = GetParam();
  auto weights = (Eigen::MatrixXd(2, 3) << 1, 2, 3, 4, 5, 6).finished();
  InputLayer layer(params.inputs, weights);

  layer.CalculateDeltasPerStep(params.index, params.deltaWeightedSum);

  ASSERT_EQ(params.expectedState, layer.GetState());
}

INSTANTIATE_TEST_SUITE_P(
    InputLayerTest_CalculateDeltasPerStep,
    InputLayerTest_CalculateDeltasPerStep,
    ::testing::Values(
        TestParamsForCalcDeltas{Eigen::MatrixXd::Zero(3, 3), 3,
                                Eigen::MatrixXd::Zero(2, 1),
                                State::ERROR}, // Index out of bounds
        TestParamsForCalcDeltas{
            Eigen::MatrixXd::Zero(4, 4), 1, Eigen::MatrixXd::Zero(2, 2),
            State::ERROR}, // Mismatch between inputs rows and deltaSum cols
        TestParamsForCalcDeltas{Eigen::MatrixXd::Zero(3, 3), 1,
                                Eigen::MatrixXd::Zero(1, 1), State::ERROR}
        // Mismatch between delta rows and deltaSum rows
        ));

TEST(TestCalculateDeltasPerStep, GivenValidInputs) {
  auto inputs = (Eigen::MatrixXd(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
  auto weights = (Eigen::MatrixXd(2, 3) << 1, 2, 3, 4, 5, 6).finished();
  InputLayer layer(inputs, weights);
  auto deltaWeightedSum = (Eigen::MatrixXd(2, 1) << 1, 2).finished();

  layer.CalculateDeltasPerStep(1, deltaWeightedSum);

  auto expected = (Eigen::MatrixXd(2, 3) << 4, 5, 6, 8, 10, 12).finished();
  ASSERT_TRUE(layer.m_deltaWeights.isApprox(expected));
  ASSERT_EQ(State::NORMAL, layer.GetState());
}

TEST(TestUpdateWeightsAndBias, GivenValidInputs) {
  auto inputs = (Eigen::MatrixXd(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
  auto weights = (Eigen::MatrixXd(2, 3) << 1, 2, 3, 4, 5, 6).finished();
  InputLayer layer(inputs, weights);
  layer.m_deltaWeights = (Eigen::MatrixXd(2, 3) << 1, 2, 3, 4, 5, 6).finished();

  layer.UpdateWeightsAndBias(2.0);

  auto expected = (Eigen::MatrixXd(2, 3) << -1, -2, -3, -4, -5, -6).finished();
  ASSERT_TRUE(layer.m_weightMatrix.isApprox(expected));
}
} // namespace rnn::model
