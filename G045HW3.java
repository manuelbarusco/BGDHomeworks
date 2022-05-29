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
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end-start) + " ms");

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

        long startTimeR1, startTimeR2, endTimeR1, endTimeR2;
        startTimeR1 = System.currentTimeMillis();

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

        //--------------------- ROUND 2 ---------------------------

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k+z)*L); //gather the coreset for avoiding lazy evaluation
        elems.addAll(coreset.collect());
        //System.out.println(elems.toString());
        endTimeR1 = System.currentTimeMillis();

        //
        // ****** ADD YOUR CODE
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        // ****** Return the final solution
        //

        startTimeR2 = System.currentTimeMillis();

        ArrayList<Vector> P = new ArrayList<>((k+z)*L);     // arrayList of points
        ArrayList<Long> W = new ArrayList<>((k+z)*L);       // arrayList of weights

        // separate the Tuple2<Vector, Long> and add key-value pairs to P and W, respectively
        for(Tuple2<Vector,Long> e : elems){
            P.add(e._1());
            W.add(e._2());
        }//for

        // compute matrix of distances
        calculateDistancesMatrix(P);

        ArrayList<Vector> centers = SeqWeightedOutliers(P,W,k,z,2);

        endTimeR2 = System.currentTimeMillis();
        System.out.println("Initial guess = " + initialGuess);
        System.out.println("Final guess = " + finalGuess);
        System.out.println("Number of guesses = " + nGuesses);
        System.out.println("Time Round 1: " + (endTimeR1 - startTimeR1) + " ms");
        System.out.println("Time Round 2: " + (endTimeR2 - startTimeR2) + " ms");

        return centers;
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

    /**
     * SeqWeightedOutliers
     * @param P point set
     * @param W integer weight set
     * @param k number of centers
     * @param z number of outliers
     * @param alpha coefficient used by the algorithm
     * @return the set of centers found
     */
    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, double alpha) {
        rInitialization(k, z);

        double r= initialGuess;
        nGuesses=1;

        while (true) {
            long Wz = 0;

            for (long w : W) {
                Wz += w;
            }//for

            HashSet<Integer> Z = new HashSet<>(); //Z: set of indexes of uncovered points

            for(int i=0;i<P.size();i++) {
                Z.add(i);
            }//for

            ArrayList<Vector> S = new ArrayList<>();  //S: set of centers, for now it's empty

            while (S.size() < k && Wz > 0) {
                long max = 0;
                Vector new_center = null;

                for (int i = 0; i < P.size(); i++) {
                    if (Z.contains(i)) {
                        long ball_weight = ball_weight(Z, (1 + 2 * alpha) * r, i, W);
                        if (ball_weight > max) {
                            max = ball_weight;
                            new_center = P.get(i);
                        }//if
                    }//if
                }//for

                S.add(new_center);
                ArrayList<Integer> ball = ball(Z, (3 + 4 * alpha) * r, P.indexOf(new_center));

                for (Integer y : ball) {
                    Z.remove(y);
                    Wz -= W.get(y); //y's weight
                }//for
            }//while

            if (Wz <= z) {
                finalGuess = r;
                return S;
            }else if(S.size()==k){
                r *= 2;
                nGuesses++;
            }//if-else

        }//while
    }//SeqWeightedOutliers


    /**
     * Function to compute the matrix of distances
     * @param inputPoints vector of input points
     * it set the static matrix field distances
     */
    public static void calculateDistancesMatrix(ArrayList<Vector> inputPoints){
        int size= inputPoints.size();
        distances = new double [size][size];
        for (int i=0; i<size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j)
                    distances[i][j] = 0.0;
                else {
                    distances[i][j] = Math.sqrt(Vectors.sqdist(inputPoints.get(i), inputPoints.get(j)));
                    distances[j][i] = distances[i][j];      //copy the value in the symmetric part of the matrix
                }//if-else
            }//for
        }//for
    }//calculateDistancesMatrix

    /**
     * function for r parameter initialization
     * @param k number of centers
     * @param z number of outliers
     */
    private static void rInitialization(int k, int z){
        double minDistance= distances[0][1];
        for(int i=0;i<distances.length && i<k+z+1; i++)
            for(int j=0;j<distances[0].length && j<k+z+1; j++)
                if(i!=j && distances[i][j]< minDistance)
                    minDistance = distances[i][j];
        initialGuess= minDistance/2;
    } //rInitialization


    /**
     * ball_weight calculation
     * @param Z indexes of uncovered points in the pointset
     * @param r radius of the ball
     * @param index_center index of the center
     * @param W set of points weights
     * @return ball weight of radius r
     */
    private static long ball_weight(HashSet<Integer> Z, double r, int index_center, ArrayList<Long> W){
        long ball_weight=0; //the center contributes to the ball weight count
        for(int j=0;j<distances[0].length; j++){
            if(Z.contains(j) && distances[index_center][j] <= r ) {
                ball_weight += W.get(j);
            }//if
        }//for
        return ball_weight;
    }//ball_weight


    /**
     * ball calculation
     * @param Z indexes of uncovered points in the pointset
     * @param r radius of the ball
     * @param center_index  index of the center
     * @return ball of radius r and center i th point components
     */
    private static ArrayList<Integer> ball(HashSet<Integer> Z, double r, int center_index){
        ArrayList<Integer> ball=new ArrayList<>();
        for(int j=0;j<distances[0].length; j++) {
            if (Z.contains(j) && distances[center_index][j] <= r)
                ball.add(j);
        }//for
        return ball;
    }//ball


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

        return points.mapPartitions(partition -> {
            //compute distances between each point in the partion and obtained centers

            //distances of every point in the partition to the nearest center
            ArrayList<Double> distances = new ArrayList<>();

            while(partition.hasNext()) {
                Vector point= partition.next();
                double minDistance = -1;
                double distance = 0;
                for (Vector center : centers) {
                    distance = euclidean(point, center);
                    if(minDistance == -1)
                        minDistance = distance;
                    else if(distance < minDistance)
                        minDistance = distance;
                }//for
                distances.add(minDistance);
            }
            //System.out.println(distances.toString());
            return distances.iterator();
        }).top(z+1).get(z);

    }//computeObjective

}