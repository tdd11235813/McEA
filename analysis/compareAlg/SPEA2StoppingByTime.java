import org.uma.jmetal.algorithm.multiobjective.spea2.SPEA2;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.problem.ConstrainedProblem;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.Solution;

import java.util.List;

/**
 * This class shows a version of SPEA2 having a stopping condition depending on run-time
 *
 * @author Eric Starke <estarke@htw-dresden.de>
 */
@SuppressWarnings("serial")
public class SPEA2StoppingByTime<S extends Solution<?>> extends SPEA2<S> {
  private long initComputingTime ;
  private long thresholdComputingTime ;
  private boolean stoppingCondition ;
  /**
   * Constructor
   */
  public SPEA2StoppingByTime(Problem<S> problem, int populationSize,
                              long maxComputingTime,
                              CrossoverOperator<S> crossoverOperator, MutationOperator<S> mutationOperator,
                              SelectionOperator<List<S>, S> selectionOperator) {
    super(problem, 0, populationSize, crossoverOperator, mutationOperator,
        selectionOperator, null);

    initComputingTime = System.currentTimeMillis() ;
    stoppingCondition = false ;
    thresholdComputingTime = maxComputingTime ;
  }

  @Override
  protected void initProgress() {
    iterations = 1;
  }

  @Override protected void updateProgress() {
  }

  @Override protected List<S> evaluatePopulation(List<S> population) {
    int index = 0 ;

    while ((index < population.size()) && !stoppingCondition) {
      if (getProblem() instanceof ConstrainedProblem) {
        getProblem().evaluate(population.get(index));
        ((ConstrainedProblem<S>) getProblem()).evaluateConstraints(population.get(index));
      } else {
        getProblem().evaluate(population.get(index));
      }

      if ((System.currentTimeMillis() - initComputingTime) > thresholdComputingTime) {
        stoppingCondition = true ;
      } else {
        index ++ ;
      }
    }

    return population;
  }

  @Override protected boolean isStoppingConditionReached() {
    return stoppingCondition ;
  }

  @Override public String getName() {
    return "SPEA2" ;
  }

  @Override public String getDescription() {
    return "SPEA2" ;
  }
}
