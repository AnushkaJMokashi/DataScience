var inputfile=sc.textFile("input.txt")
var counts=inputfile.flatMap(line=>line.split(" ")).map(word=>(word,1)).reduceByKey(_+_);
// counts.toDebugString
// counts.cache()
// counts.saveAsTextFile("output")
println()
counts.collect().foreach(println)
println()
println("Your text is : ")
inputfile.collect().foreach(println)