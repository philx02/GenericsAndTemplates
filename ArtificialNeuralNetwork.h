#pragma once

#include <array>
#include <cstddef>

template< typename T, std::size_t InputLayerSize, std::size_t HiddenLayerSize, std::size_t OutputLayerSize, typename _ActivationFunction, typename _OutputFunction >
class ArtificialNeuralNetwork
{
public:
  typedef T value_type;
  typedef _ActivationFunction ActivationFunction;
  typedef _OutputFunction OutputFunction;
  typedef std::array< std::array< T, InputLayerSize >, HiddenLayerSize > InputToHiddenWeights;
  typedef std::array< std::array< T, HiddenLayerSize >, OutputLayerSize > HiddenToOutputWeights;

  ArtificialNeuralNetwork(ActivationFunction iActivationFunction, OutputFunction iOutputFunction)
    : mActivationFunction(std::move(iActivationFunction))
    , mOutputFunction(std::move(iOutputFunction))
  {
  }

  inline std::array< value_type, OutputLayerSize > compute(const std::array< value_type, InputLayerSize > &iInput) const
  {
    return computeLayer< OutputLayerSize >(mHiddenToOutput, computeLayer< HiddenLayerSize >(mInputToHidden, iInput, mActivationFunction), mOutputFunction);
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

private:
  template< std::size_t ToLayerSize, std::size_t FromLayerSize >
  std::array< value_type, ToLayerSize > computeLayer(const std::array< std::array< value_type, FromLayerSize >, ToLayerSize > &wFromToTo, const std::array< value_type, FromLayerSize > &iFromLayerInput, ActivationFunction &&iActivationFunction) const
  {
    std::array< value_type, ToLayerSize > wToLayerOutput;
    for (std::size_t wToNeuronIndex = 0; wToNeuronIndex < ToLayerSize; ++wToNeuronIndex)
    {
      value_type wSum = 0;
      for (std::size_t wFromNeuronIndex = 0; wFromNeuronIndex < FromLayerSize; ++wFromNeuronIndex)
      {
        wSum += wFromToTo[wToNeuronIndex][wFromNeuronIndex] * iFromLayerInput[wFromNeuronIndex];
      }
      wToLayerOutput[wToNeuronIndex] = iActivationFunction(wSum);
    }
    return wToLayerOutput;
  }

  ActivationFunction mActivationFunction;
  OutputFunction mOutputFunction;
  InputToHiddenWeights mInputToHidden;
  HiddenToOutputWeights mHiddenToOutput;
};

template< typename T, typename WeightType, typename InitializeFunction >
void initializeWeights(WeightType &ioWeights, InitializeFunction &&iInitializeFunction)
{
  for (auto &&iLayer : ioWeights)
  {
    for (auto &&wWeight : iLayer)
    {
      wWeight = iInitializeFunction(); 
    }
  }
}

template< typename T, std::size_t InputLayerSize, std::size_t HiddenLayerSize, std::size_t OutputLayerSize, typename ActivationFunction, typename OutputFunction, typename WeightInitializerFunction >
ArtificialNeuralNetwork< T, InputLayerSize, HiddenLayerSize, OutputLayerSize, ActivationFunction, OutputFunction > createAndInitializeArtificialNeuralNetwork(ActivationFunction &&iActivationFunction, OutputFunction &&iOutputFunction, WeightInitializerFunction &&iWeightInitializer)
{
  typedef ArtificialNeuralNetwork< T, InputLayerSize, HiddenLayerSize, OutputLayerSize, ActivationFunction, OutputFunction > Ann;
  Ann wAnn(std::move(iActivationFunction), std::move(iOutputFunction));
  wAnn.applyToInputToHiddenWeights([&](Ann::InputToHiddenWeights &ioWeights)
  {
    initializeWeights< Ann::value_type, Ann::InputToHiddenWeights >(ioWeights, iWeightInitializer);
  });
  wAnn.applyToHiddenToOutputWeights([&](Ann::HiddenToOutputWeights &ioWeights)
  {
    initializeWeights< Ann::value_type, Ann::HiddenToOutputWeights >(ioWeights, iWeightInitializer);
  });
  return wAnn;
}
