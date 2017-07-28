class Constants {
  // config constants
  public static final double mutationProbability = 0.01;
  public static final int generations = 100;
  public static final int popWidth = 30;
  public static final int parameters = 20;
  public static final int objectives = 3;
  public static final int dtlzNum = 7;
  public static final String outFile = "nsgaii";
  
  // derived constants
  public static final int populationSize = popWidth * (popWidth + 1);
  public static final int maxEvaluations = generations * populationSize;
}
