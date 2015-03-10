#pragma once

#include <vector>

template< typename T, typename _Individual, typename _IndividualInstantiator >
class GeneticAlgorithm
{
public:
  typedef T value_type;
  typedef _Individual Individual;
  typedef _IndividualInstantiator IndividualInstantiator;
  struct RatedIndividual;
  typedef std::vector< RatedIndividual > Population;
  typedef std::vector< typename Population::iterator > RankedPopulationProxy;

  struct RatedIndividual
  {
    RatedIndividual(IndividualInstantiator &&iIndividualInstantiator) : mIndividual(iIndividualInstantiator()), mScore(0) {}
    RatedIndividual(const Individual &iIndividual) : mIndividual(iIndividual), mScore(0) {}
    Individual mIndividual;
    value_type mScore;
  };

  GeneticAlgorithm(size_t iInitialPopulationSize, IndividualInstantiator &&iIndividualInstantiator)
    : mMaxPopulationSize(iInitialPopulationSize)
    , mMinPopulationSize(mMaxPopulationSize * 3 / 4)
  {
    mPopulation.reserve(iInitialPopulationSize);
    mRankedPopulationProxy.reserve(iInitialPopulationSize);
    for (size_t wIndex = 0; wIndex < iInitialPopulationSize; ++wIndex)
    {
      mPopulation.emplace_back(iIndividualInstantiator());
      mRankedPopulationProxy.emplace_back(mPopulation.end() - 1);
    }
  }

  template< typename FitnessEvaluation, typename ReproductionFunction >
  inline void runGenerations(size_t iNumberOfGenerations, FitnessEvaluation &&iFitnessEvaluation, ReproductionFunction &&iReproductionFunction)
  {
    runGenerations(iNumberOfGenerations, iFitnessEvaluation, iReproductionFunction, [](auto, auto) { return true; });
  }

  template< typename FitnessEvaluation, typename ReproductionFunction, typename PostGenerationEvaluation >
  inline void runGenerations(size_t iNumberOfGenerations, FitnessEvaluation &&iFitnessEvaluation, ReproductionFunction &&iReproductionFunction, PostGenerationEvaluation &&iPostGenerationEvaluation)
  {
    runGenerations(iNumberOfGenerations, iFitnessEvaluation, iReproductionFunction, iPostGenerationEvaluation, [](auto, auto) {});
  }

  template< typename FitnessEvaluation, typename ReproductionFunction, typename PreGenerationSetup, typename PostGenerationEvaluation >
  inline void runGenerations(size_t iNumberOfGenerations, FitnessEvaluation &&iFitnessEvaluation, ReproductionFunction &&iReproductionFunction, PostGenerationEvaluation &&iPostGenerationEvaluation, PreGenerationSetup &&iPreGenerationSetup)
  {
    for (size_t wGeneration = 0; wGeneration < iNumberOfGenerations && runOneGeneration(iFitnessEvaluation, iReproductionFunction, iPostGenerationEvaluation, iPreGenerationSetup); ++wGeneration);
  }

  template< typename FitnessEvaluation, typename ReproductionFunction, typename PreGenerationSetup, typename PostGenerationEvaluation >
  inline bool runOneGeneration(FitnessEvaluation &&iFitnessEvaluation, ReproductionFunction &&iReproductionFunction, PostGenerationEvaluation &&iPostGenerationEvaluation, PreGenerationSetup &&iPreGenerationSetup)
  {
    iPreGenerationSetup(begin(), end());
    iFitnessEvaluation(mPopulation.begin(), mPopulation.end());
    std::sort(begin(), end(), [](auto &&iLeft, auto &&iRight)
    {
      return iLeft->mScore > iRight->mScore;
    });
    std::for_each(begin(), begin() + (mMaxPopulationSize - mMinPopulationSize), [&](auto &&iRatedIndividual)
    {
      *iRatedIndividual = iReproductionFunction(begin(), end());
    });
    return iPostGenerationEvaluation(begin(), end());
  }

  inline typename RankedPopulationProxy::iterator begin()
  {
    return std::begin(mRankedPopulationProxy);
  }

  inline typename RankedPopulationProxy::iterator end()
  {
    return std::end(mRankedPopulationProxy);
  }

  inline typename RankedPopulationProxy::value_type & front()
  {
    return *std::begin(mRankedPopulationProxy);
  }

  inline typename RankedPopulationProxy::value_type & back()
  {
    return *std::rbegin(mRankedPopulationProxy);
  }

  inline typename RankedPopulationProxy::size_type size()
  {
    return mRankedPopulationProxy.size();
  }

private:
  size_t mMaxPopulationSize;
  size_t mMinPopulationSize;
  Population mPopulation;
  RankedPopulationProxy mRankedPopulationProxy;
};

template< typename T, typename Individual, typename IndividualInstantiator >
GeneticAlgorithm< T, Individual, IndividualInstantiator > createGeneticAlgorithm(size_t iInitialPopulationSize, IndividualInstantiator &&iIndividualInstantiator)
{
  typedef GeneticAlgorithm< T, Individual, IndividualInstantiator > Ga;
  return Ga(iInitialPopulationSize, std::move(iIndividualInstantiator));
}
