//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.problem.impl.AbstractDoubleProblem;
import org.uma.jmetal.runner.AbstractAlgorithmRunner;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.ProblemUtils;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ1;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ2;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ3;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ4;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ5;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ6;
import org.uma.jmetal.problem.multiobjective.dtlz.DTLZ7;
import org.uma.jmetal.util.fileoutput.SolutionListOutput;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import org.uma.jmetal.util.JMetalException;

import java.io.FileNotFoundException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.List;
import java.util.Arrays;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.io.IOException;
import java.lang.String;

/**
 * Class to configure and run the NSGA-II algorithm
 *
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 */
public class NewNSGAIIRunner extends AbstractAlgorithmRunner {
  static private ArrayList<Class<? extends AbstractDoubleProblem>> problems = new ArrayList<Class<? extends AbstractDoubleProblem>>();

  /**
   * @param args Command line arguments.
   * @throws JMetalException
   * @throws FileNotFoundException
   * Invoking command:
    java org.uma.jmetal.runner.multiobjective.NSGAIIRunner problemName [referenceFront]
   */
  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws JMetalException, FileNotFoundException {

    problems.add(DTLZ1.class);
    problems.add(DTLZ2.class);
    problems.add(DTLZ3.class);
    problems.add(DTLZ4.class);
    problems.add(DTLZ5.class);
    problems.add(DTLZ6.class);
    problems.add(DTLZ7.class);

    Problem<DoubleSolution> problem;
    Algorithm<List<DoubleSolution>> algorithm;
    CrossoverOperator<DoubleSolution> crossover;
    MutationOperator<DoubleSolution> mutation;
    SelectionOperator<List<DoubleSolution>, DoubleSolution> selection;
    int runNumber = 0;
    String writeFolder = ".";

    if (args.length == 1) {
      writeFolder = args[0];
    } else if (args.length == 2) {
      writeFolder = args[0];
      runNumber = Integer.parseInt(args[1]);
    }

    Class<?>[] cArg = new Class<?>[2];
    cArg[0] = Integer.class;
    cArg[1] = Integer.class;

    try {
      problem = (Problem<DoubleSolution>)problems.get(Constants.dtlzNum - 1).getDeclaredConstructor(cArg).newInstance(Constants.parameters, Constants.objectives) ;
    } catch (Exception e) {
      throw new JMetalException("Problem instance cannot be build.", e) ;
    }


    int numberOfCrossovers = 1;
    crossover = new DoubleNPointCrossover(numberOfCrossovers);

    mutation = new RealUniformMutation(Constants.mutationProbability) ;

    selection = new BinaryTournamentSelection<DoubleSolution>(
        new RankingAndCrowdingDistanceComparator<DoubleSolution>());

    algorithm = new NSGAIIBuilder<DoubleSolution>(problem, crossover, mutation)
        .setSelectionOperator(selection)
        .setMaxEvaluations(Constants.maxEvaluations)
        .setPopulationSize(Constants.populationSize)
        .build() ;

    AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm)
        .execute() ;

    List<DoubleSolution> population = algorithm.getResult() ;
    long computingTime = algorithmRunner.getComputingTime() ;

    JMetalLogger.logger.info("Total execution time: " + computingTime + "ms");

    String fileName = writeFolder + "/" + Constants.outFile + 
      "_g" + Constants.generations +
      "_pw" + Constants.popWidth +
      "_p" + Constants.parameters +
      "_r0_t1_vs0" +
      "_dt" + Constants.dtlzNum + "_" + runNumber;

    try {
      BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(fileName + ".obj"));
      int numberOfObjectives = population.get(0).getNumberOfObjectives();
      for (int i = 0; i < population.size(); i++) {
        for (int j = 0; j < numberOfObjectives; j++) {
          bufferedWriter.write(
              String.format("%1$1.10f", population.get(i).getObjective(j)) + "\t");
        }
        bufferedWriter.newLine();
      }
      bufferedWriter.close();
    } catch (IOException e) {
      throw new JMetalException("Error writing data ", e) ;
    }    

    writeInfos(fileName, computingTime);
  }

  private static void writeInfos(String fileName, long runtime) {
    List<String> lines = Arrays.asList(
        "name:\t" + fileName + ".info",
        "runtime:\t" + runtime + " ms",
        "dtlz_problem:\t" + Constants.dtlzNum,
        "threads:\t1",
        "generations:\t" + Constants.generations,
        "pop_width:\t" + Constants.popWidth,
        "pop_size:\t" + Constants.populationSize,
        "param_count:\t" + Constants.parameters,
        "objectives:\t" + Constants.objectives,
        "neighborhood:\t0",
        "mutation_prob:\t" + Constants.mutationProbability);

    Path file = Paths.get(fileName + ".info");
    try {
      Files.write(file, lines, Charset.forName("UTF-8"));
    } catch(IOException e) {
      throw new JMetalException("Info File cannot be written.", e) ;
    }
  }
}
