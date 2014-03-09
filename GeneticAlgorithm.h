#pragma once

#include <vector>
#include <amp.h>

template< typename T, typename _Individual, typename _IndividualInstantiator >
class GeneticAlgorithm
{
public:
  typedef T value_type;
  typedef _Individual Individual;
  typedef _IndividualInstantiator IndividualInstantiator;
  struct RatedIndividual;
  typedef std::vector< RatedIndividual > Population;

  struct RatedIndividual
  {
    RatedIndividual(IndividualInstantiator &&iIndividualInstantiator) : mIndividual(iIndividualInstantiator()), mScore(0.0) {}
    RatedIndividual(const Individual &iIndividual) : mIndividual(iIndividual), mScore(0.0) {}
    Individual mIndividual;
    value_type mScore;
  };

  GeneticAlgorithm(std::size_t iInitialPopulationSize, IndividualInstantiator &&iIndividualInstantiator)
    : mMaxPopulationSize(iInitialPopulationSize)
    , mMinPopulationSize(mMaxPopulationSize * 3 / 4)
  {
    mPopulation.reserve(iInitialPopulationSize);
    for (std::size_t wIndex = 0; wIndex < iInitialPopulationSize; ++wIndex)
    {
      mPopulation.emplace_back(iIndividualInstantiator);
    }
  }

  template< typename FitnessFunction, typename ReproductionFunction, typename PreGenerationSetup, typename PostGenerationEvaluation >
  inline void runGenerations(std::size_t iNumberOfGenerations, FitnessFunction &&iFitnessFunction, ReproductionFunction &&iReproductionFunction, PreGenerationSetup &&iPreGenerationSetup, PostGenerationEvaluation &&iPostGenerationEvaluation)
  {
    for (std::size_t wGeneration = 0; wGeneration < iNumberOfGenerations && runOneGeneration(iFitnessFunction, iReproductionFunction, iPreGenerationSetup, iPostGenerationEvaluation); ++wGeneration);
  }

  template< typename FitnessFunction, typename ReproductionFunction, typename PreGenerationSetup, typename PostGenerationEvaluation >
  inline bool runOneGeneration(FitnessFunction &&iFitnessFunction, ReproductionFunction &&iReproductionFunction, PreGenerationSetup &&iPreGenerationSetup, PostGenerationEvaluation &&iPostGenerationEvaluation)
  {
    iPreGenerationSetup(begin(), end());
    //for (auto &&iRatedIndividual : mPopulation)
    Concurrency::parallel_for_each(begin(), end(), [&](decltype(*begin()) && iRatedIndividual)
    {
      iRatedIndividual.mScore = iFitnessFunction(iRatedIndividual.mIndividual);
    });
    std::sort(begin(), end(), [](decltype(*begin()) && iLeft, decltype(*begin()) && iRight)
    {
      return iLeft.mScore > iRight.mScore;
    });
    while (mPopulation.size() > mMinPopulationSize)
    {
      mPopulation.pop_back();
    }
    while (mPopulation.size() < mMaxPopulationSize)
    {
      mPopulation.emplace_back(iReproductionFunction(begin(), end()));
    }
    return iPostGenerationEvaluation(begin(), end());
  }

  inline typename Population::iterator begin()
  {
    return std::begin(mPopulation);
  }

  inline typename Population::iterator end()
  {
    return std::end(mPopulation);
  }

private:
  std::size_t mMaxPopulationSize;
  std::size_t mMinPopulationSize;
  Population mPopulation;
};

template< typename T, typename Individual, typename IndividualInstantiator >
GeneticAlgorithm< T, Individual, IndividualInstantiator > createGeneticAlgorithm(std::size_t iInitialPopulationSize, IndividualInstantiator &&iIndividualInstantiator)
{
  typedef GeneticAlgorithm< T, Individual, IndividualInstantiator > Ga;
  return Ga(iInitialPopulationSize, std::move(iIndividualInstantiator));
}
