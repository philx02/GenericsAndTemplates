#pragma once

#include <array>
#include <cstddef>
#include <algorithm>
#include <xutility>

namespace polyview
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
    return mArray[iIndex];
  }
  inline const T & operator[](size_t iIndex) const
  {
    return mArray[iIndex];
  }

  inline auto begin()
  {
    return std::begin(mArray);
  }
  inline auto begin() const
  {
    return std::begin(mArray);
  }

  inline auto end()
  {
    return std::end(mArray);
  }
  inline auto end() const
  {
    return std::end(mArray);
  }

  // restrict(amp)

  inline T & operator[](size_t iIndex) restrict(amp)
  {
    return reinterpret_cast< T * >(&mArray)[iIndex];
  }
  inline const T & operator[](size_t iIndex) const restrict(amp)
  {
    return reinterpret_cast< const T * >(&mArray)[iIndex];
  }

  inline auto begin() restrict(amp)
  {
    return std::addressof(reinterpret_cast< T * >(&mArray)[0]);
  }
  inline auto begin() const restrict(amp)
  {
    return std::addressof(reinterpret_cast< T * >(&mArray)[0]);
  }

  inline auto end() restrict(amp)
  {
    return std::addressof(reinterpret_cast< T * >(&mArray)[Size]);
  }
  inline auto end() const restrict(amp)
  {
    return std::addressof(reinterpret_cast< T * >(&mArray)[Size]);
  }

private:
  std::array< T, Size > mArray;
};
}

namespace std
{

// unrestricted

template< typename T, size_t Size >
inline auto begin(polyview::array< T, Size > &iArray)
{
  return iArray.begin();
}
template< typename T, size_t Size >
inline auto end(polyview::array< T, Size > &iArray)
{
  return iArray.end();
}

// restrict(amp)

template< typename T, size_t Size >
inline auto begin(polyview::array< T, Size > &iArray) restrict(amp)
{
  return iArray.begin();
}
template< typename T, size_t Size >
inline auto end(polyview::array< T, Size > &iArray) restrict(amp)
{
  return iArray.end();
}
}

template< typename T, size_t InputLayerSize, size_t HiddenLayerSize, size_t OutputLayerSize, typename _ActivationPolicy, typename _OutputPolicy >
class ArtificialNeuralNetwork
{
public:
  typedef T value_type;
  typedef _ActivationPolicy ActivationPolicy;
  typedef _OutputPolicy OutputPolicy;
  typedef polyview::array< T, InputLayerSize > Input;
  typedef polyview::array< Input, HiddenLayerSize > InputToHiddenWeights;
  typedef polyview::array< polyview::array< T, HiddenLayerSize >, OutputLayerSize > HiddenToOutputWeights;
  typedef polyview::array< T, OutputLayerSize > Output;

  typedef polyview::array< T, HiddenLayerSize > HiddenBiases;
  typedef Output OutputBiases;

  inline static size_t inputLayerSize()
  {
    return InputLayerSize;
  }
  inline static size_t hiddenLayerSize()
  {
    return HiddenLayerSize;
  }
  inline static size_t outputLayerSize()
  {
    return OutputLayerSize;
  }
  inline static size_t inputToHiddenWeightSize()
  {
    return InputLayerSize * HiddenLayerSize;
  }
  inline static size_t hiddenToOutputWeightSize()
  {
    return HiddenLayerSize * OutputLayerSize;
  }
  inline static size_t totalWeightSize()
  {
    return inputToHiddenWeightSize() + hiddenToOutputWeightSize();
  }
  inline static size_t totalBiasesSize()
  {
    return HiddenLayerSize + OutputLayerSize;
  }
    
  inline Output compute(const Input &iInput) const restrict(cpu, amp)
  {
    auto wActivationFunction = [](const T &iInput) restrict(cpu, amp)
    {
      return ActivationPolicy::compute(iInput);
    };
    auto wOutputFunction = [](const T &iInput) restrict(cpu, amp)
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
  template< size_t ToLayerSize, size_t FromLayerSize, typename ActivationFunction >
  inline polyview::array< value_type, ToLayerSize > computeLayer(const polyview::array< polyview::array< value_type, FromLayerSize >, ToLayerSize > &wWeightMatrix, const polyview::array< value_type, ToLayerSize > &wBiases, const polyview::array< value_type, FromLayerSize > &iFromLayerInput, ActivationFunction &&iActivationFunction) const restrict(cpu, amp)
  {
    polyview::array< value_type, ToLayerSize > wToLayerOutput;
    for (size_t wToNeuronIndex = 0; wToNeuronIndex < ToLayerSize; ++wToNeuronIndex)
    {
      value_type wSum = 0;
      for (size_t wFromNeuronIndex = 0; wFromNeuronIndex < FromLayerSize; ++wFromNeuronIndex)
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

template< typename T, size_t InputLayerSize, size_t HiddenLayerSize, size_t OutputLayerSize, typename ActivationPolicy, typename OutputPolicy, typename InitializeFunction >
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

template< typename T, size_t Span, size_t Offset >
class LogisticFunctionPolicy
{
public:
  static inline T compute(const T &iInput) restrict(cpu, amp)
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
  static inline T compute(const T &iInput) restrict(cpu, amp)
  {
    return iInput;
  }
};

template< typename T, size_t LowerLimit, size_t HigherLimit >
class SaturatedLinearFunctionPolicy
{
public:
  static inline T compute(const T &iInput) restrict(cpu, amp)
  {
    static const auto wLowerLimit = static_cast< T >(LowerLimit);
    static const auto wHigherLimit = static_cast< T >(HigherLimit);
    return iInput > wHigherLimit ? wHigherLimit : (iInput < wLowerLimit ? wLowerLimit : iInput);
  }
};
