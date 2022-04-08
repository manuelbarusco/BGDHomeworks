import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

/*
GROUP: 045
COMPONENTS:
- Barusco Manuel
- Gregori Andrea
- Rampon Riccardo
 */

public class G045HW1 {

    //Comparator class for sorting productPopularity1 and productPopularity2 JavaPairRDDs
    public static class ProductPopularityComparator implements Comparator<Tuple2<String,Long>>{

        @Override
        public int compare(Tuple2<String, Long> o1, Tuple2<String, Long> o2) {
            return o1._1().compareTo(o2._1());
        }
    }

    public static void main(String[] args) throws IllegalArgumentException, IOException {

        //simple check of the command line arguments
        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: K H S dataset_path");
        }

        // SPARK SETUP
        SparkConf conf = new SparkConf(true).setAppName("HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // COMMAND LINE INPUT READING
        int K = Integer.parseInt(args[0]);      // Read number of partitions
        int H = Integer.parseInt(args[1]);      // Read number H
        String S = args[2];                     // Read country S

        // Read input file and subdivide it into K random partitions (cache() forces the system to compute the RDD)
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        // SETTING GLOBAL VARIABLES
        long numRows, numProductCustumer;
        Random randomGenerator = new Random();

        //*************** TASK 1 ***************

        numRows = rawData.count();      // number of elements of the rawData RDD
        System.out.println("Number of rows = " + numRows);

        //definition of all the JavaPairRDD that we will use
        JavaPairRDD<String, Integer> productCustomer;
        JavaPairRDD<String, Long> productPopularity1;
        JavaPairRDD<String, Long> productPopularity2;
        JavaPairRDD<Long, String> rate;


        //*************** TASK 2 ***************
        productCustomer = rawData
                    .filter( //filtering transactions with country and quantity parameters
                            (transaction) -> {
                                String[] tokens = transaction.split(",");
                                if (S.equalsIgnoreCase("all")){
                                    //Checking quantity > 0
                                    return Integer.parseInt(tokens[3]) > 0;
                                }
                                //Checking quantity > 0 and Country = S
                                return Integer.parseInt(tokens[3]) > 0 && tokens[7].equalsIgnoreCase(S);
                            }
                    )
                    .mapToPair( //Extracting ProductID and CustomerID from each transaction
                            (transaction) -> {
                                String[] tokens = transaction.split(",");
                                Tuple2<Tuple2<String, Integer>, Integer> tuple = new Tuple2<>(new Tuple2<String, Integer>(tokens[1],Integer.parseInt(tokens[6])), 1);
                                return tuple;
                            }
                    )
                    .groupByKey()
                    .mapToPair( //Removing the integer value 1
                            (t) ->{
                                Tuple2<String, Integer> pair = new Tuple2<>(t._1()._1(), t._1()._2());
                                return pair;
                            }
                    );
        numProductCustumer = productCustomer.count();   // number of elements of productCustomer
        System.out.println("Product-Customer Pairs = " + numProductCustumer);


        //*************** TASK 3 ***************

        productPopularity1 = productCustomer //MAP PHASE (R1) EMPTY
                .mapPartitionsToPair((productCustomerPair) -> { //REDUCE PHASE (R1)

                    //execute counts of product-popularity in each partition
                    HashMap<String, Long> counts = new HashMap<>();
                    while(productCustomerPair.hasNext()) {
                        Tuple2<String, Integer> tuple= productCustomerPair.next();
                        counts.put(tuple._1(), 1L+ counts.getOrDefault(tuple._1(), 0L));
                    } //while

                    //create the output for the next round
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    } //for

                    return pairs.iterator();
                }) //MAP PHASE (R2) EMPTY
                .groupByKey().mapValues((it) -> { //REDUCE PHASE (R2)
                    long sum=0L;
                    for(long c : it){
                        sum+=c;
                    }
                    return sum;
                });


        //*************** TASK 4 ***************

        productPopularity2 = productCustomer
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
                .reduceByKey((x,y) -> x+y);


        //*************** TASK 6 ***************

        if(H==0) {
            //task6: print all the pairs in productPopularity1 in lexicographic order
            System.out.println("productPopularity1:");
            ArrayList<Tuple2<String, Long>> productPopularity1List = new ArrayList<>(productPopularity1.collect());
            productPopularity1List.sort(new ProductPopularityComparator());
            for (Tuple2<String, Long> line : productPopularity1List) {
                System.out.print("Product: " + line._1() + " Popularity: " + line._2() + "; ");
            }//for

            //task6: print all the pairs in productPopularity2 in lexicographic order
            System.out.println("\nproductPopularity2:");
            ArrayList<Tuple2<String, Long>> productPopularity2List = new ArrayList<>(productPopularity2.collect());
            productPopularity2List.sort(new ProductPopularityComparator());
            for (Tuple2<String, Long> line : productPopularity2List) {
                System.out.print("Product: " + line._1() + " Popularity: " + line._2() + "; ");
            }//for
        }

        //*************** TASK 5 ***************

        if(H>0) {
            //task5: save in a list and prints the ProductID and Popularity of the H products with highest Popularity
            rate = productPopularity1.mapToPair((pp1) -> pp1.swap()).sortByKey(false);
            System.out.println("Top "+H+ " Products and their Popularities");
            ArrayList<Tuple2<Long, String>> rateList = new ArrayList<>(rate.take(H));
            for (Tuple2<Long, String> ppr : rateList) {
                System.out.print("Product " + ppr._2() + " Popularity " + ppr._1() + "; ");
            }//for
        }
    }
}
