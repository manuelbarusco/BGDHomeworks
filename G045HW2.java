import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import javax.sound.midi.SysexMessage;
import java.io.IOException;

/*
GROUP: 045
COMPONENTS:
- Barusco Manuel
- Gregori Andrea
- Rampon Riccardo
 */

public class G045HW2 {

    private static double[][] distances;
    private static double initialGuess;
    private static double finalGuess;
    private static int nGuesses;

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
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    //******** Distances Matrix calculation ********

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
                    distances[i][j] = 0.0f;     //zero in float
                else {
                    distances[i][j] = Math.sqrt(Vectors.sqdist(inputPoints.get(i), inputPoints.get(j)));
                    distances[j][i] = distances[i][j];      //copy the value in the symmetric part of the matrix
                }
            }
        }
    }

    /**
     * function for distance initialization
     * @param k number of centers
     * @param z number of outliers
     */
    private static void rInitialization(int k, int z){
        double minDistance= distances[0][1];
        for(int i=0;i<distances.length && i<k+z+1; i++)
            for(int j=0;j<distances[0].length && j<k+z+1; j++)
                if(i!=j && distances[i][j]< minDistance)
                    minDistance = distances[i][j];
        initialGuess=minDistance/2;
    } //rInitialization

    /**
     * ball_weight calculation
     * @param Z indexes of uncovered points in the pointset
     * @param r radius of the ball
     * @param index_center index of the center
     * @param W set of points weights
     * @return ball weight of radius r
     */
    private static long ball_weight(ArrayList<Integer> Z, double r, int index_center, ArrayList<Long> W){
        long ball_weight=0; //the center contributes to the ball weight count
        for(int j=0;j<distances[0].length; j++){
            if(Z.contains(j) && distances[index_center][j] <= r )
                ball_weight+=W.get(j);
        }
        return ball_weight;
    }

    /**
     * ball calculation
     * @param Z indexes of uncovered points in the pointset
     * @param r radius of the ball
     * @param center_index  index of the center
     * @return ball of radius r and center i th point components
     */
    private static ArrayList<Integer> ball(ArrayList<Integer> Z, double r, int center_index){
        ArrayList<Integer> ball=new ArrayList<>();
        for(int j=0;j<distances[0].length; j++) {
            if (Z.contains(j) && distances[center_index][j] <= r)
                ball.add(j);
        }
        return ball;
    }

    /**
     * SeqWeightedOutliers
     * @param P point set
     * @param W integer weight set
     * @param k number of centers
     * @param z number of outliers
     * @param alpha coefficient used by the algorithm
     * @return the set of centers
     */
    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, double alpha) {
        rInitialization(k, z);
        double r= initialGuess;
        nGuesses=1;

        while (true) {
            long Wz = 0;
            for (long w : W)
                Wz += w;
            ArrayList<Integer> Z = new ArrayList<>(); //Z: set of indexes of uncovered points
            for(int i=0;i<P.size();i++)
                Z.add(i);

            ArrayList<Vector> S = new ArrayList<>();  //S: set of centers, for now it's empty

            while (S.size() < k && Wz > 0) {
                long max = 0;

                Vector new_center = null; //create an empty vector
                for (int i = 0; i < P.size(); i++) {
                    if (Z.contains((Integer) i)) {
                        long ball_weight = ball_weight(Z, (1 + 2 * alpha) * r, i, W);
                        //System.out.println("Iterazione:" + i + " Ball_W: " + ball_weight);
                        if (ball_weight > max) {
                            max = ball_weight;
                            new_center = P.get(i);
                        }
                    }
                }

                //System.out.println("Centro da aggiungere: " + new_center);

                S.add(new_center);

                ArrayList<Integer> ball = ball(Z, (3 + 4 * alpha) * r, P.indexOf(new_center));
                /*System.out.println("Palla da rimuovere:");
                for (Integer i : ball)
                    System.out.println(i);*/
                for (Integer y : ball) {
                    Z.remove(y);
                    Wz -= W.get(y); //peso di y
                }
            }
            /*System.out.println("Uncovered points:");
               for (Integer i : Z)
                   System.out.println(i);*/
            if (Wz <= z && S.size()==k) {
                finalGuess = r;
                return S;
            } else if(S.size()==k){
                //System.out.println("nuova guess");
                r *= 2;
                nGuesses++;
                //System.out.println(r);
            }
            //System.out.println("Wz:" + Wz);
        }
    }

    /** ComputeObjective(P,S,z)
     * @param P set of input points
     * @param S set of centers returned by SeqWeightedOutliers
     * @param z number of outliers
     * @return the largest distance between points in P and centers in S, discarding the last $z
     */
    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z){
        Double[] local_distances = new Double[(P.size() * S.size()) - S.size()];
        double distance;
        int index = 0;

        for(Vector p_i : P) {
            for(Vector center_i : S) {
                if(!p_i.equals(center_i)){
                    distance = Math.sqrt(Vectors.sqdist(p_i, center_i));
                    //System.out.println("Distance (" + p_i + "," + center_i + ") = " + distance);
                    local_distances[index] = distance;
                    index++;
                }//if
            }//foreach
        }//foreach

        // sort all distances compute before
        java.util.Arrays.sort(local_distances);
        //System.out.println("local_distances" + java.util.Arrays.toString(local_distances));

        return local_distances[(local_distances.length-1)-z];
    }//ComputeObjective

    public static void main(String[] args) throws IllegalArgumentException, IOException {

        //simple check of the command line arguments
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: dataset_path k z");
        }

        // Command line input reading
        String data_path = args[0];             // Read dataset path
        int k = Integer.parseInt(args[1]);      // Read k (the number of centers)
        int z = Integer.parseInt(args[2]);      // Read z (the number of allowed outliers)

        //Read points in the dataset
        ArrayList<Vector> inputPoints = readVectorsSeq(data_path);

        //Initializing weights with all ones
        ArrayList<Long> weights = new ArrayList<>(inputPoints.size());
        for (int i=0; i<inputPoints.size(); i++)
            weights.add(1L);

        //******* COMPUTE THE MATRIX OF DISTANCES *******
        calculateDistancesMatrix(inputPoints);

        //Plot the distance matrix

        /*for (int i=0; i<inputPoints.size(); i++){
            for (int j=0; j<inputPoints.size(); j++) {
                System.out.print(distances[i][j] + " ");
            }
            System.out.println();
        } /*

        //Plot the points

        for (int i=0; i<inputPoints.size(); i++){
            System.out.println("" + inputPoints.get(i));
        }*/

        long startTime= System.currentTimeMillis();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints,weights,k,z,0);
        long endTime=System.currentTimeMillis();

        long exTime= endTime-startTime;

        // OUTPUT OF THE SYSTEM
        System.out.println("Input size n = "+inputPoints.size());
        System.out.println("Number of centers k = "+k);
        System.out.println("Number of outliers z = "+z);
        System.out.println("Initial guess = "+initialGuess);
        System.out.println("Final guess = "+finalGuess);
        System.out.println("Number of guesses = "+nGuesses);
        System.out.println("Objective function = ");
        System.out.println("Time of SeqWeightedOutliers = "+exTime);

    }
}