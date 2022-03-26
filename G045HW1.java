import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.HashSet;

public class G045HW1 {

    public static void main(String[] args) throws IOException {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: K H S dataset_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");     //Reduce the warning

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        int K = Integer.parseInt(args[0]);      // Read number of partitions
        int H = Integer.parseInt(args[1]);      // Read number H
        String S = args[2];                     // Read country S

        // Read input file and subdivide it into K random partitions (cache() forces the system to compute the RDD)
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long numRows, numProductCustumer;
        numRows = rawData.count();      // number of elements of the rawData RDD
        System.out.println("Number of rows = " + numRows);

        JavaPairRDD<String, Integer> productCustumer;

        productCustumer = rawData
                .filter(
                        (transaction) -> {
                            String[] tokens = transaction.split(",");
                            if (S.equalsIgnoreCase("all")){
                                if (Integer.parseInt(tokens[3])>0)      //Checking quantity > 0
                                    return true;
                                else
                                    return false;
                            }
                            if (Integer.parseInt(tokens[3])>0 && tokens[7].equalsIgnoreCase(S))     //Checking quantity > 0 and Country = S
                                return true;
                            else
                                return false;
                        }
                )
                .mapToPair(
                        (transaction) -> {
                            String[] tokens = transaction.split(",");
                            Tuple2<Tuple2<String, Integer>, Integer> tuple = new Tuple2<Tuple2<String, Integer>, Integer>(new Tuple2<String, Integer>(tokens[1],Integer.parseInt(tokens[6])), 1);
                            return tuple;
                        }
                )
                .reduceByKey((x, y) -> x+y)
                .mapToPair(
                        (t) ->{
                          Tuple2<String, Integer> pair = new Tuple2<String, Integer>(t._1()._1(), t._1()._2());
                          return pair;
                        }
                );
        numProductCustumer = productCustumer.count();
        System.out.println("Product-Customer Pairs =" + numProductCustumer);

        // SOLO PER TEST DA TOGLIERE!!
        for(Tuple2<String, Integer> line:productCustumer.collect()){
            System.out.println("* "+line);
        }

    }
}
