#pragma once

#include <array>
#include <cstddef>
#include <algorithm>

template< typename T, std::size_t InputLayerSize, std::size_t HiddenLayerSize, std::size_t OutputLayerSize, typename _ActivationFunction, typename _OutputFunction >
class ArtificialNeuralNetwork
{
public:
  typedef T value_type;
  typedef _ActivationFunction ActivationFunction;
  typedef _OutputFunction OutputFunction;
  typedef std::array< T, InputLayerSize > Input;
  typedef std::array< Input, HiddenLayerSize > InputToHiddenWeights;
  typedef std::array< std::array< T, HiddenLayerSize >, OutputLayerSize > HiddenToOutputWeights;
  typedef std::array< T, OutputLayerSize > Output;

  typedef std::array< T, HiddenLayerSize > HiddenBiases;
  typedef Output OutputBiases;

  ArtificialNeuralNetwork(ActivationFunction iActivationFunction, OutputFunction iOutputFunction)
    : mActivationFunction(std::move(iActivationFunction))
    , mOutputFunction(std::move(iOutputFunction))
  {
  }

  inline Output compute(const Input &iInput) const
  {
    return computeLayer< OutputLayerSize >(mHiddenToOutput, mOutputBiases, computeLayer< HiddenLayerSize >(mInputToHidden, mHiddenBiases, iInput, mActivationFunction), mOutputFunction);
  }

  template< typename Function >
  inline void applyToInputToHiddenWeights(Function &&iFunction)
  {
    iFunction(mInputToHidden);
  }
  template< typename Function >
  inline void applyToInputToHiddenWeights(Function &&iFunction) const
  {
    iFunction(mInputToHidden);
  }

  template< typename Function >
  inline void applyToHiddenToOutputWeights(Function &&iFunction)
  {
    iFunction(mHiddenToOutput);
  }
  template< typename Function >
  inline void applyToHiddenToOutputWeights(Function &&iFunction) const
  {
    iFunction(mHiddenToOutput);
  }

  template< typename Function >
  inline void applyToHiddenBiases(Function &&iFunction)
  {
    iFunction(mHiddenBiases);
  }
  template< typename Function >
  inline void applyToHiddenBiases(Function &&iFunction) const
  {
    iFunction(mHiddenBiases);
  }

  template< typename Function >
  inline void applyToOutputBiases(Function &&iFunction)
  {
    iFunction(mOutputBiases);
  }
  template< typename Function >
  inline void applyToOutputBiases(Function &&iFunction) const
  {
    iFunction(mOutputBiases);
  }

private:
  template< std::size_t ToLayerSize, std::size_t FromLayerSize >
  inline std::array< value_type, ToLayerSize > computeLayer(const std::array< std::array< value_type, FromLayerSize >, ToLayerSize > &wWeightMatrix, const std::array< value_type, ToLayerSize > &wBiases, const std::array< value_type, FromLayerSize > &iFromLayerInput, ActivationFunction &&iActivationFunction) const
  {
    std::array< value_type, ToLayerSize > wToLayerOutput;
    for (std::size_t wToNeuronIndex = 0; wToNeuronIndex < ToLayerSize; ++wToNeuronIndex)
    {
      value_type wSum = 0;
      for (std::size_t wFromNeuronIndex = 0; wFromNeuronIndex < FromLayerSize; ++wFromNeuronIndex)
      {
        wSum += wWeightMatrix[wToNeuronIndex][wFromNeuronIndex] * iFromLayerInput[wFromNeuronIndex];
      }
      wSum -= wBiases[wToNeuronIndex];
      wToLayerOutput[wToNeuronIndex] = iActivationFunction(wSum);
    }
    return wToLayerOutput;
  }

  ActivationFunction mActivationFunction;
  OutputFunction mOutputFunction;
  InputToHiddenWeights mInputToHidden;
  HiddenToOutputWeights mHiddenToOutput;
  HiddenBiases mHiddenBiases;
  OutputBiases mOutputBiases;
};

template< typename T, typename WeightType, typename InitializeFunction >
inline void initializeWeights(WeightType &ioWeights, InitializeFunction &&iInitializeFunction)
{
  for (auto &&iLayer : ioWeights)
  {
    std::generate(std::begin(iLayer), std::end(iLayer), iInitializeFunction);
  }
}

template< typename T, std::size_t InputLayerSize, std::size_t HiddenLayerSize, std::size_t OutputLayerSize, typename ActivationFunction, typename OutputFunction, typename InitializeFunction >
ArtificialNeuralNetwork< T, InputLayerSize, HiddenLayerSize, OutputLayerSize, ActivationFunction, OutputFunction > createAndInitializeArtificialNeuralNetwork(ActivationFunction &&iActivationFunction, OutputFunction &&iOutputFunction, InitializeFunction &&iInitializeFunction)
{
  typedef ArtificialNeuralNetwork< T, InputLayerSize, HiddenLayerSize, OutputLayerSize, ActivationFunction, OutputFunction > Ann;
  Ann wAnn(std::move(iActivationFunction), std::move(iOutputFunction));
  wAnn.applyToInputToHiddenWeights([&](Ann::InputToHiddenWeights &ioWeights)
  {
    initializeWeights< Ann::value_type, Ann::InputToHiddenWeights >(ioWeights, iInitializeFunction);
  });
  wAnn.applyToHiddenToOutputWeights([&](Ann::HiddenToOutputWeights &ioWeights)
  {
    initializeWeights< Ann::value_type, Ann::HiddenToOutputWeights >(ioWeights, iInitializeFunction);
  });
  wAnn.applyToHiddenBiases([&](Ann::HiddenBiases &ioBiases)
  {
    std::generate(std::begin(ioBiases), std::end(ioBiases), iInitializeFunction);
  });
  wAnn.applyToOutputBiases([&](Ann::OutputBiases &ioBiases)
  {
    std::generate(std::begin(ioBiases), std::end(ioBiases), iInitializeFunction);
  });
  return wAnn;
}
