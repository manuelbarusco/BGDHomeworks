import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import scala.collection.Iterable;

import java.util.*;

/*
GROUP: 045
COMPONENTS:
- Barusco Manuel
- Gregori Andrea
- Rampon Riccardo
 */

public class G045HW3 {

    private static double[][] distances; //matrix of the distances
    private static double initialGuess;  //initial guess of r
    private static double finalGuess;    //final guess of r
    private static int nGuesses;         //number of guesses made by the algorithm

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x-> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();



        // ----- Print input parameters
        //System.out.println("File : " + filename);
        //System.out.println("Number of points N = " + N);
        //System.out.println("Number of centers k = " + k);
        //System.out.println("Number of outliers z = " + z);
        //System.out.println("Number of partitions L = " + L);
         //System.out.println("Time to read from file: " + (end-start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);
        System.out.println("Size solution: " + solution.size());

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end-start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method euclidean: distance function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method MR_kCenterOutliers: MR algorithm for k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers (JavaRDD<Vector> points, int k, int z, int L){

        long startTime, endTime;
        startTime = System.currentTimeMillis();

        //------------- ROUND 1 ---------------------------

        JavaRDD<Tuple2<Vector,Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());

            ArrayList<Vector> centers = kCenterFFT(partition, k+z+1);
            //System.out.println("centers: " + centers.toString());

            ArrayList<Long> weights = computeWeights(partition, centers);
            //System.out.println("weight: " + weights.toString());

            ArrayList<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i =0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1

        //------------- ROUND 2 ---------------------------

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k+z)*L); //gather the coreset
        elems.addAll(coreset.collect());
        //System.out.println(elems.toString());
        endTime = System.currentTimeMillis();

        System.out.println("Time Round 1:" + (startTime - endTime));

        //
        // ****** ADD YOUR CODE
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        // ****** Return the final solution
        //

        startTime = System.currentTimeMillis();

        ArrayList<Vector> P = new ArrayList<>((k+z)*L);
        ArrayList<Long> W = new ArrayList<>((k+z)*L);

        // separate the Tuple2<Vector, Long> and add key-value pairs to P and W, respectively
        for(Tuple2<Vector,Long> e : elems){
            P.add(e._1());
            W.add(e._2());
        }//for

        ArrayList<Vector> solution = SeqWeightedOutliers(P,W,k,z,2);

        endTime = System.currentTimeMillis();
        System.out.println("Time Round 2:" + (startTime - endTime));

        return solution;
    }//MR_kCenterOutliers

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT (ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius =0;

        for (int iter=1; iter<k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers){
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i = 0; i < points.size(); ++i)
        {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j)
            {
                if(euclidean(points.get(i),centers.get(j)) < tmp)
                {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    /** ComputeObjective(P,S,z)
     * @param points set of input points
     * @param centers set of centers returned by SeqWeightedOutliers
     * @param z number of outliers
     * @return the largest distance between points in P and centers in S, by not considering the last $z
     */
    public static double computeObjective (JavaRDD<Vector> points, ArrayList<Vector> centers, int z) {

         double objValue = points.map(point -> {
            //compute distances between each points and centers
            ArrayList<Double> distances = new ArrayList<>();
            for(Vector center : centers){
                distances.add(euclidean(point, center));
            }//for
            //System.out.println(distances.toString());
            return distances.iterator();
        })
        .sortBy(dist -> dist,false,0) //sorting distances
        .map(dist -> { //remove $z outliers
            for(int i = 0; i<z; i++){
                dist.remove();
            }
            return dist;
        }).collect().get(0);




        return objValue;
    }//computeObjective

}