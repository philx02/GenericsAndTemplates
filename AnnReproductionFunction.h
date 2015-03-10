#pragma once

#include <cassert>

template< typename NumericalType, typename Ann, typename RandomNumberGenerator >
void mutateWeights(Ann &iAnn, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wWeightMutationDistribution = std::uniform_real_distribution< NumericalType >(-1.0, 1.0);
  static auto wWeightRangeDistribution = std::uniform_int_distribution< size_t >(0, Ann::totalWeightSize() - 1);
  static auto wTotalNumberOfWeightMutations = ([]() { return std::max(Ann::totalWeightSize() / 20, 1u); })();
  for (auto wIter = 0u; wIter < wTotalNumberOfWeightMutations; ++wIter)
  {
    auto wIndexToMutate = wWeightRangeDistribution(iRandomNumberGenerator);
    if (wIndexToMutate < Ann::inputToHiddenWeightSize())
    {
      iAnn.applyToInputToHiddenWeights([&](Ann::InputToHiddenWeights &iWeights)
      {
        iWeights[wIndexToMutate / Ann::inputLayerSize()][wIndexToMutate % Ann::inputLayerSize()] += wWeightMutationDistribution(iRandomNumberGenerator);
      });
    }
    else
    {
      wIndexToMutate -= Ann::inputToHiddenWeightSize();
      iAnn.applyToHiddenToOutputWeights([&](Ann::HiddenToOutputWeights &iWeights)
      {
        iWeights[wIndexToMutate / Ann::hiddenLayerSize()][wIndexToMutate % Ann::hiddenLayerSize()] += wWeightMutationDistribution(iRandomNumberGenerator);
      });
    }
  }
}

template< typename NumericalType, typename Ann, typename RandomNumberGenerator >
void mutateBiases(Ann &iAnn, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wBiasMutationDistribution = std::uniform_real_distribution< NumericalType >(-1.0, 1.0);
  static auto wBiasRangeDistribution = std::uniform_int_distribution< size_t >(0, Ann::totalBiasesSize() - 1);
  static auto wTotalNumberOfBiasesMutations = ([]() { return std::max(Ann::totalBiasesSize() / 20, 1u); })();
  for (auto wIter = 0u; wIter < wTotalNumberOfBiasesMutations; ++wIter)
  {
    auto wIndexToMutate = wBiasRangeDistribution(iRandomNumberGenerator);
    if (wIndexToMutate < Ann::hiddenLayerSize())
    {
      iAnn.applyToHiddenBiases([&](Ann::HiddenBiases &iBiases)
      {
        iBiases[wIndexToMutate] += wBiasMutationDistribution(iRandomNumberGenerator);
      });
    }
    else
    {
      iAnn.applyToOutputBiases([&](Ann::OutputBiases &iBiases)
      {
        iBiases[wIndexToMutate - Ann::hiddenLayerSize()] += wBiasMutationDistribution(iRandomNumberGenerator);
      });
    }
  }
}

template< typename NumericalType, typename Ann, typename RandomNumberGenerator >
Ann cloneAndMutate(const Ann &iSource, RandomNumberGenerator &iRandomNumberGenerator)
{
  auto wOffspring = iSource;
  mutateWeights< NumericalType >(wOffspring, iRandomNumberGenerator);
  mutateBiases< NumericalType >(wOffspring, iRandomNumberGenerator);
  return wOffspring;
}

template< typename NumericalType, typename Individual, typename IteratorProxy, typename RandomNumberGenerator >
Individual reproductionFunction(IteratorProxy iBegin, IteratorProxy iEnd, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wNormalDistribution = std::normal_distribution<>(0, 100);
  size_t wPopulationSize = iEnd - iBegin;
  assert(wPopulationSize > 0);
  size_t wIndividualIndex = 0;
  do
  {
    wIndividualIndex = static_cast< size_t >(std::abs(wNormalDistribution(iRandomNumberGenerator)));
  } while (wIndividualIndex >= wPopulationSize);

  return cloneAndMutate< NumericalType >(iBegin[wIndividualIndex]->mIndividual, iRandomNumberGenerator);
}
