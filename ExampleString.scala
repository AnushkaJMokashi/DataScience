object ExampleString{
    def main(args: Array[String]){
        val text : String = "You are reading a scala Program";
        println("Value is :"+text)
         println("Enter a string:")
        
        val inputString = scala.io.StdIn.readLine()
        println("You enteres this: "+inputString)
    }
}

ExampleString.main(Array())