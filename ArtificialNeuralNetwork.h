#pragma once

#include <array>
#include <cstddef>
#include <algorithm>
#include <xutility>

namespace smuds
{

template< typename T, size_t Size >
class array
{
public:
  typedef T value_type;
  typedef decltype(Size) size_type;
  typedef value_type * pointer;
  typedef pointer iterator;
  typedef const pointer const_iterator;

  // unrestricted

  inline T & operator[](size_t iIndex)
  {
    return mElements[iIndex];
  }
  inline const T & operator[](size_t iIndex) const
  {
    return mElements[iIndex];
  }

  iterator begin()
  {
    return std::addressof(mElements[0]);
  }
  const value_type * const begin() const
  {
    return std::addressof(mElements[0]);
  }

  iterator end()
  {
    return std::addressof(mElements[Size]);
  }
  const value_type * const end() const
  {
    return std::addressof(mElements[Size]);
  }

  // restrict(amp)

  inline T & operator[](size_t iIndex) restrict(amp)
  {
    return mElements[iIndex];
  }
  inline const T & operator[](size_t iIndex) const restrict(amp)
  {
    return mElements[iIndex];
  }

  iterator begin() restrict(amp)
  {
    return std::addressof(mElements[0]);
  }
  const_iterator begin() const restrict(amp)
  {
    return std::addressof(mElements[0]);
  }

  iterator end() restrict(amp)
  {
    return std::addressof(mElements[Size]);
  }
  const_iterator end() const restrict(amp)
  {
    return std::addressof(mElements[Size]);
  }

private:
  T mElements[Size];
};
}

namespace std
{
template< typename T, size_t Size >
typename smuds::array< T, Size >::iterator begin(smuds::array< T, Size > &iArray) restrict(amp)
{
  return iArray.begin();
}
template< typename T, size_t Size >
typename smuds::array< T, Size >::iterator end(smuds::array< T, Size > &iArray) restrict(amp)
{
  return iArray.end();
}
template< typename T, size_t Size >
typename smuds::array< T, Size >::iterator begin(smuds::array< T, Size > &iArray)
{
  return iArray.begin();
}
template< typename T, size_t Size >
typename smuds::array< T, Size >::iterator end(smuds::array< T, Size > &iArray)
{
  return iArray.end();
}
}

template< typename T, std::size_t InputLayerSize, std::size_t HiddenLayerSize, std::size_t OutputLayerSize, typename _ActivationPolicy, typename _OutputPolicy >
class ArtificialNeuralNetwork
{
public:
  typedef T value_type;
  typedef _ActivationPolicy ActivationPolicy;
  typedef _OutputPolicy OutputPolicy;
  typedef smuds::array< T, InputLayerSize > Input;
  typedef smuds::array< Input, HiddenLayerSize > InputToHiddenWeights;
  typedef smuds::array< smuds::array< T, HiddenLayerSize >, OutputLayerSize > HiddenToOutputWeights;
  typedef smuds::array< T, OutputLayerSize > Output;

  typedef smuds::array< T, HiddenLayerSize > HiddenBiases;
  typedef Output OutputBiases;

  inline Output compute(const Input &iInput) const restrict(amp)
  {
    auto wActivationFunction = [](const T &iInput) restrict(amp)
    {
      return ActivationPolicy::compute(iInput);
    };
    auto wOutputFunction = [](const T &iInput) restrict(amp)
    {
      return OutputPolicy::compute(iInput);
    };
    return computeLayer< OutputLayerSize >(mHiddenToOutput, mOutputBiases, computeLayer< HiddenLayerSize >(mInputToHidden, mHiddenBiases, iInput, wActivationFunction), wOutputFunction);
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
  template< std::size_t ToLayerSize, std::size_t FromLayerSize, typename ActivationFunction >
  inline smuds::array< value_type, ToLayerSize > computeLayer(const smuds::array< smuds::array< value_type, FromLayerSize >, ToLayerSize > &wWeightMatrix, const smuds::array< value_type, ToLayerSize > &wBiases, const smuds::array< value_type, FromLayerSize > &iFromLayerInput, ActivationFunction &&iActivationFunction) const restrict(amp)
  {
    smuds::array< value_type, ToLayerSize > wToLayerOutput;
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

template< typename T, std::size_t InputLayerSize, std::size_t HiddenLayerSize, std::size_t OutputLayerSize, typename ActivationPolicy, typename OutputPolicy, typename InitializeFunction >
ArtificialNeuralNetwork< T, InputLayerSize, HiddenLayerSize, OutputLayerSize, ActivationPolicy, OutputPolicy > createAndInitializeArtificialNeuralNetwork(InitializeFunction &&iInitializeFunction)
{
  typedef ArtificialNeuralNetwork< T, InputLayerSize, HiddenLayerSize, OutputLayerSize, ActivationPolicy, OutputPolicy > Ann;
  Ann wAnn;
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

// Typical policies

template< typename T, std::size_t Span, std::size_t Offset >
class LogisticFunctionPolicy
{
public:
  static inline T compute(const T &iInput) restrict(amp)
  {
    static const auto wSpan = static_cast< T >(Span);
    static const auto wOffset = static_cast< T >(Offset);
    static const auto wOne = static_cast< T >(1);
    return wSpan / (wOne + std::exp(-iInput)) + wOffset;
  }
};

template< typename T >
class LinearFunctionPolicy
{
public:
  static inline T compute(const T &iInput) restrict(amp)
  {
    return iInput;
  }
};

template< typename T, std::size_t LowerLimit, std::size_t HigherLimit >
class SaturatedLinearFunctionPolicy
{
public:
  static inline T compute(const T &iInput) restrict(amp)
  {
    static const auto wLowerLimit = static_cast< T >(LowerLimit);
    static const auto wHigherLimit = static_cast< T >(HigherLimit);
    return iInput > wHigherLimit ? wHigherLimit : (iInput < wLowerLimit ? wLowerLimit : iInput);
  }
};
