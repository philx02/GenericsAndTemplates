#pragma once

#include <cassert>

template< typename NumericalType, typename Ann, typename RandomNumberGenerator >
void mutateWeights(Ann &iAnn, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wWeightMutationDistribution = std::uniform_real_distribution< NumericalType >(-1.0, 1.0);
  static auto wWeightRangeDistribution = std::uniform_int_distribution<>(0, sTotalWeightSize - 1);
  for (auto wIter = 0u; wIter < sTotalNumberOfWeightMutations; ++wIter)
  {
    auto wIndexToMutate = wWeightRangeDistribution(iRandomNumberGenerator);
    if (wIndexToMutate < sInputToHiddenWeightSize)
    {
      iAnn.applyToInputToHiddenWeights([&](Ann::InputToHiddenWeights &iWeights)
      {
        iWeights[wIndexToMutate / sInputLayerSize][wIndexToMutate % sInputLayerSize] += wWeightMutationDistribution(iRandomNumberGenerator);
      });
    }
    else
    {
      wIndexToMutate -= sInputToHiddenWeightSize;
      iAnn.applyToHiddenToOutputWeights([&](Ann::HiddenToOutputWeights &iWeights)
      {
        iWeights[wIndexToMutate / sHiddenLayerSize][wIndexToMutate % sHiddenLayerSize] += wWeightMutationDistribution(iRandomNumberGenerator);
      });
    }
  }
}

template< typename NumericalType, typename Ann, typename RandomNumberGenerator >
void mutateBiases(Ann &iAnn, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wBiasMutationDistribution = std::uniform_real_distribution< NumericalType >(-1.0, 1.0);
  static auto wBiasRangeDistribution = std::uniform_int_distribution<>(0, sTotalBiasesSize - 1);
  for (auto wIter = 0u; wIter < sTotalNumberOfBiasesMutations; ++wIter)
  {
    auto wIndexToMutate = wBiasRangeDistribution(iRandomNumberGenerator);
    if (wIndexToMutate < sHiddenLayerSize)
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
        iBiases[wIndexToMutate - sHiddenLayerSize] += wBiasMutationDistribution(iRandomNumberGenerator);
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

template< typename NumericalType, typename Individual, typename Iterator, typename RandomNumberGenerator >
Individual reproductionFunction(Iterator iBegin, Iterator iEnd, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wNormalDistribution = std::normal_distribution<>(0, 100);
  std::size_t wPopulationSize = iEnd - iBegin;
  assert(wPopulationSize > 0);
  std::size_t wIndividualIndex = 0;
  do
  {
    wIndividualIndex = static_cast< std::size_t >(std::abs(wNormalDistribution(iRandomNumberGenerator)));
  } while (wIndividualIndex >= wPopulationSize);

  return cloneAndMutate< NumericalType >(iBegin[wIndividualIndex].mIndividual, iRandomNumberGenerator);
}
