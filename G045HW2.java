import java.io.*;
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
        int size = inputPoints.size();

        //Initializing weights with all ones
        ArrayList<Long> weights = new ArrayList<>(inputPoints.size());
        for (int i=0; i<size; i++)
            weights.add(1L);

        //******* COMPUTE THE MATRIX OF DISTANCES *******
        Float distances[][] = new Float[inputPoints.size()][inputPoints.size()];
        for (int i=0; i<size; i++){
            for (int j=0; j<size; j++){
                if (i == j)
                    distances[i][j] = 0.0f;     //zero in float
                else {
                    distances[i][j] = (float) Math.sqrt(Vectors.sqdist(inputPoints.get(i), inputPoints.get(j)));
                    distances[j][i] = distances[i][j];      //copy the value in the symmetric part of the matrix
                }
            }
        }

        //Plot the distance matrix
        /*
        for (int i=0; i<size; i++){
            for (int j=0; j<size; j++) {
                System.out.print(distances[i][j] + " ");
            }
            System.out.println();
        }*/

        //Plot the points
        /*
        for (int i=0; i<size; i++){
            System.out.println("" + inputPoints.get(i));
        }*/
    }


    //******** Input reading methods ********

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
}
