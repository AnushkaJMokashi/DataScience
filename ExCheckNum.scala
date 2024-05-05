object ExCheckNum{
    def main(args : Array[String]){
        println("Enter a number:")
        val num = scala.io.StdIn.readInt()
        println(num)
        // var num= (-100)
        if(num==0){
            println("Number is 0");
        }
        else if(num>0){
            println("Num is positive")
        }
        else{
            println("Num is negative")
        }
    }
}

ExCheckNum.main(Array());