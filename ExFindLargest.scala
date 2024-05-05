object ExFindLargest{
    def main(args: Array[String]){
        //var num1=20;
        //var num2=30;
        var num1 = scala.io.StdIn.readInt()
        var num2 = scala.io.StdIn.readInt()
        if(num1>num2){
            println("largest is:"+num1);
        }
        else {
            println("Largest number is:"+num2);
        }
    }
}

ExFindLargest.main(Array())
