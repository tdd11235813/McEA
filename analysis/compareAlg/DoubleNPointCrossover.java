import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.solution.DoubleSolution;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.ArrayUtils;

/**
 */
public class DoubleNPointCrossover implements CrossoverOperator<DoubleSolution> {
    private final JMetalRandom RNG = JMetalRandom.getInstance();
    private final double probability;
    private final int crossovers;

    public DoubleNPointCrossover(double probability, int crossovers) {
        if(probability < 0.0) throw new JMetalException("Probability can't be negative");
        if(crossovers < 1) throw new JMetalException("Number of crossovers is less than one");
        this.probability = probability;
        this.crossovers = crossovers;
    }

    public DoubleNPointCrossover(int crossovers){
        this.crossovers = crossovers;
        this.probability = 1.0;
    }

    @Override
    public int getNumberOfParents() {
        return 2;
    }

    public double getCrossoverProbability() {
        return probability;
    }

    @Override
    public List<DoubleSolution> execute(List<DoubleSolution> s){
        if(getNumberOfParents() != s.size()){
            throw new JMetalException("Point Crossover requires + " + getNumberOfParents() + " parents, but got " + s.size());
        }
        if(RNG.nextDouble() < probability) {
            return doCrossover(s);
        } else {
            return s;
        }
    }

    private List<DoubleSolution> doCrossover(List<DoubleSolution> s){
        DoubleSolution mom = s.get(0);
        DoubleSolution dad = s.get(1);

        if (mom.getNumberOfVariables() != dad.getNumberOfVariables()){
            throw new JMetalException("The 2 parents doesn't have the same number of variables");
        }
        if (mom.getNumberOfVariables() < crossovers){
            throw new JMetalException("The number of crossovers is higher than the number of variables");
        }

        int[] crossoverPoints = new int[crossovers];
        for (int i = 0; i < crossoverPoints.length; i++) {
            crossoverPoints[i] = RNG.nextInt(0, mom.getNumberOfVariables()-1);
        }
        DoubleSolution girl = (DoubleSolution)mom.copy();
        DoubleSolution boy = (DoubleSolution)dad.copy();
        boolean swap = false;

        for (int i = 0; i < mom.getNumberOfVariables(); i++) {
            if(swap){
                boy.setVariableValue(i, mom.getVariableValue(i));
                girl.setVariableValue(i,dad.getVariableValue(i));

            }

            if(ArrayUtils.contains(crossoverPoints,i)){
                swap = !swap;
            }
        }
        List<DoubleSolution> result = new ArrayList<>();
        result.add(girl);
        result.add(boy);
        return result;
    }

}

