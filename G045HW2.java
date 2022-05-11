import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.io.IOException;

/*
GROUP: 045
COMPONENTS:
- Barusco Manuel
- Gregori Andrea
- Rampon Riccardo
 */

public class G045HW2 {

    private static double[][] distances; //matrix of the distances
    private static double initialGuess;  //initial guess of r
    private static double finalGuess;    //final guess of r
    private static int nGuesses;         //number of guesses made by the algorithm

    //******** Input reading methods ********

    /**
     * @param str string contains vector component
     * @return Vector representation of the vector contained in str
     */
    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }//for
        return Vectors.dense(data);
    }//strToVector

    /**
     * @param filename file name
     * @return ArrayList of points (vectors)
     * @throws IOException
     */
    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }//if
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }//readVectorsSeq

    //******** Helper methods ********

    /**
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

    //******** Homework methods ********

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

    /** ComputeObjective(P,S,z)
     * @param P set of input points
     * @param S set of centers returned by SeqWeightedOutliers
     * @param z number of outliers
     * @return the largest distance between points in P and centers in S, by not considering the last $z
     */
    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z){
        double[] local_distances = new double[P.size()];

        // Initializing the distances vector with all -1
        for(int i=0; i<local_distances.length;i++) {
            local_distances[i] = -1;
        }//for

        double distance;
        int index = 0;

        for(Vector p_i : P) {
            for(Vector center_i : S) {
                if(p_i.equals(center_i)){
                    local_distances[index]=0d;
                    break;
                } else {
                    distance = Math.sqrt(Vectors.sqdist(p_i, center_i));
                    if(local_distances[index] == -1) {
                        local_distances[index] = distance;
                    }else if(distance < local_distances[index]){
                        local_distances[index] = distance;
                    }//if-else
                }//if-else
            }//foreach
            index++;
        }//foreach

        // sort all distances computed before
        java.util.Arrays.sort(local_distances);

        // remove the $z largest distances and return the value
        return local_distances[(local_distances.length-1)-z];
    }//ComputeObjective

    //******** Main ********

    public static void main(String[] args) throws IllegalArgumentException, IOException {

        //simple check of the command line arguments
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: dataset_path k z");
        }//if

        // Command line input reading
        String data_path = args[0];             // Read dataset path
        int k = Integer.parseInt(args[1]);      // Read k (the number of centers)
        int z = Integer.parseInt(args[2]);      // Read z (the number of allowed outliers)

        //Read points in the dataset
        ArrayList<Vector> inputPoints = readVectorsSeq(data_path);

        //Initializing weights with all ones
        ArrayList<Long> weights = new ArrayList<>(inputPoints.size());
        for (int i=0; i<inputPoints.size(); i++) {
            weights.add(1L);
        }//for

        //******* COMPUTE THE MATRIX OF DISTANCES *******
        calculateDistancesMatrix(inputPoints);

        long startTime= System.currentTimeMillis();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints,weights,k,z,0);
        long endTime=System.currentTimeMillis();

        //calculate the execution time of SeqWeightedOutliers
        long exTime= endTime-startTime;

        //******* OUTPUT OF THE SYSTEM *******
        System.out.println("Input size n = "+inputPoints.size());
        System.out.println("Number of centers k = "+k);
        System.out.println("Number of outliers z = "+z);
        System.out.println("Initial guess = "+initialGuess);
        System.out.println("Final guess = "+finalGuess);
        System.out.println("Number of guesses = "+nGuesses);
        System.out.println("Objective function = "+ ComputeObjective(inputPoints,solution, z));
        System.out.println("Time of SeqWeightedOutliers = "+exTime);

    }//main

}//G045HW2