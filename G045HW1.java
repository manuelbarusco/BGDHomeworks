import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class G045HW1 {

    //Comparator class for sorting productPopularity1 and 2 JavaPairRDDs
    public static class ProductPopularityComparator implements Comparator<Tuple2<String,Long>>{

        @Override
        public int compare(Tuple2<String, Long> o1, Tuple2<String, Long> o2) {
            return o1._1().compareTo(o2._1());
        }
    }

    public static void main(String[] args) throws IOException {

        //simple check of the command line arguments
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

        JavaPairRDD<String, Integer> productCustomer;
        JavaPairRDD<String, Long> productPopularity1;
        JavaPairRDD<String, Long> productPopularity2;

        productCustomer = rawData
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

        numProductCustumer = productCustomer.count();
        System.out.println("Product-Customer Pairs =" + numProductCustumer);

        productPopularity1 = productCustomer
                .mapToPair((pc) -> {    // <-- MAP PHASE (R1) EMPTY
                    return pc;
                })
                .mapPartitionsToPair((productCustomerPair) -> { // <-- REDUCE PHASE (R1)

                    //execute counts in each partition
                    HashMap<String, Long> counts = new HashMap<>();
                    while(productCustomerPair.hasNext()) {
                        Tuple2<String, Integer> tuple= productCustomerPair.next();
                        counts.put(tuple._1(), 1L+ counts.getOrDefault(tuple._1(), 0L));
                    }

                    //create the output
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return pairs.iterator();
                })
                .mapToPair((pc) -> {    // <-- MAP PHASE (R1) EMPTY
                    return pc;
                })
                .reduceByKey((x,y) -> x+y);

        //task6: print all the pairs in productPopularity1 in lexicographic order
        System.out.println("\nproductPopularity1:");
        ArrayList<Tuple2<String, Long>> productPopularity1List = new ArrayList<>(productPopularity1.collect());
        productPopularity1List.sort(new ProductPopularityComparator());
        for(Tuple2<String, Long> line:productPopularity1List){
            System.out.println("Product: "+line._1()+" Popularity: "+line._2()+" ");
        }

        Random randomGenerator = new Random();

        productPopularity2 = productCustomer
            .mapToPair( (prodCust) -> {
                return prodCust;
            })
            .groupBy( (prodCustPair) -> randomGenerator.nextInt(K))
            .flatMapToPair((element) -> {

                //counts for each partitions
                HashMap<String, Long> counts = new HashMap<>();
                for(Tuple2<String, Integer> c : element._2()){
                    counts.put(c._1(), 1L+ counts.getOrDefault(c._1(), 0L));
                }//for

                //create output pairs
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                for(Map.Entry<String,Long> e : counts.entrySet()){
                    pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                }//for

                return pairs.iterator();
          })
          .mapToPair( pair -> {
              return pair;
          })
          .reduceByKey((x,y) -> x+y);

        //task6: print all the pairs in productPopularity2 in lexicographic order
        System.out.println("\nproductPopularity2:");
        ArrayList<Tuple2<String, Long>> productPopularity2List = new ArrayList<>(productPopularity2.collect());
        productPopularity2List.sort(new ProductPopularityComparator());
        for(Tuple2<String, Long> line:productPopularity2List){
            System.out.println("* Product: "+line._1()+" -> Popularity: "+line._2()+" ");
        }//for

    }//main
}//G045HW1
