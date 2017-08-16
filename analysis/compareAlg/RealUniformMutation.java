import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.pseudorandom.RandomGenerator;

/**
 * This class implements a uniform mutation operator.
 *
 */
@SuppressWarnings("serial")
public class RealUniformMutation implements MutationOperator<DoubleSolution> {
  private Double mutationProbability = null;
  private RandomGenerator<Double> randomGenenerator ;

  /** Constructor */
  public RealUniformMutation(double mutationProbability) {
	  this(mutationProbability, () -> JMetalRandom.getInstance().nextDouble());
  }

  /** Constructor */
  public RealUniformMutation(double mutationProbability, RandomGenerator<Double> randomGenenerator) {
    this.mutationProbability = mutationProbability ;
    this.randomGenenerator = randomGenenerator ;
  }

  public Double getMutationProbability() {
    return mutationProbability;
  }

  /* Setters */
  public void setMutationProbability(Double mutationProbability) {
    this.mutationProbability = mutationProbability;
  }

  /**
   * Perform the operation
   *
   * @param probability Mutation setProbability
   * @param solution    The solution to mutate
   */
  public void doMutation(double probability, DoubleSolution solution)  {
    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
      if (randomGenenerator.getRandomValue() < probability) {
        double rand = randomGenenerator.getRandomValue();

        rand = rand * (solution.getUpperBound(i) - solution.getLowerBound(i)) + solution.getLowerBound(i);

        solution.setVariableValue(i, rand);
      }
    }
  }

  /** Execute() method */
  @Override
  public DoubleSolution execute(DoubleSolution solution) {
    if (null == solution) {
      throw new JMetalException("Null parameter");
    }

    doMutation(mutationProbability, solution);

    return solution;
  }
}
