import Math.ActivationFunction
import Tools.DataLoader
import Tools.DataSet
import Tools.DataType
import jcuda.jcublas.JCublas
import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasPointerMode
import jcuda.runtime.JCuda
import networks.Network
import java.io.*
import kotlin.math.sqrt


val BATCH_SIZE = 1
val EPOCHS = 5
val LEARNING_RATE = 0.01

fun main(args: Array<String>)
{

    /* Initialize JCublas */
    JCublas.cublasInit()
    JCublas2.setExceptionsEnabled(true)
    JCuda.setExceptionsEnabled(true)

    JCublas2.cublasCreate(CudaFunctions.handle)
    JCublas2.cublasSetPointerMode(CudaFunctions.handle, cublasPointerMode.CUBLAS_POINTER_MODE_HOST)
    testNN()
    JCublas2.cublasDestroy(CudaFunctions.handle)

    /* Shutdown */JCublas.cublasShutdown()
}

fun testNN() {
    val loader = DataLoader("mnist_png/training")
    val dataHolder = loader.loadData(DataType.MNIST_PNG, 15.0) // val dataHolder = loadObject("save_data_holders/data_holder_mnist_15_percent_test.dh") as DataSet.DataHolder //
    //saveObject("save_data_holders/data_holder_mnist_15_percent_test.dh", dataHolder)

    val builder = Network.Builder(784, 10)
    val n = builder
        .addConvolutionalLayer(3,4,false, ActivationFunction.ReLU)
        .addConvolutionalLayer(3,4,false, ActivationFunction.ReLU)
        .addConvolutionalLayer(3,4,true, ActivationFunction.ReLU)
        .addFullyConnectedLayer(24, ActivationFunction.LeakyReLU)
        .addFullyConnectedLayer(24, ActivationFunction.LeakyReLU)
        .addFullyConnectedLayer(10, ActivationFunction.Softmax)
        .build()

    val trainer = Trainer()
    trainer.trainNetwork(n, dataHolder.training.size(), BATCH_SIZE, EPOCHS, LEARNING_RATE, 1.0, dataHolder.training)

    val tester = Tester(n, dataHolder.testing) { doubles: DoubleArray, doubles1: DoubleArray -> myAccuracyFunc(doubles, doubles1)}
    println("Average Accuracy: "+tester.test())

    saveObject("saved_networks/convnet0.nn", n)
   // n.toCSV("saved_networks/${n.getNetworkName()}")
}

fun myAccuracyFunc(output: DoubleArray, target: DoubleArray) : Double
{
    var acc = 0.0
    when(getLargestIndex(output) == getLargestIndex(target))
    {
        true -> acc = 1.0
        false -> acc = 0.0
    }

    return acc
}

fun getLargestIndex(arr: DoubleArray) : Int
{
    realAssert(arr.size > 0)
    var largestIndex = 0
    var largest = arr[0]
    for(i in arr.indices)
    {
        if(arr[i] > largest){
            largest = arr[i]
            largestIndex = i
        }
    }

    return largestIndex
}

fun realAssert(boolean: Boolean)
{
    if(!boolean)
    {
        throw Exception()
    }
}

fun saveObject(filename : String, obj: Any)
{
    //Write the family map object to a file
    val file = File("$filename")
    ObjectOutputStream(FileOutputStream(file)).use{ it -> it.writeObject(obj)}
}

fun loadObject(filename : String) : Any {

    val file = File("$filename")
    var obj : Any? = null
    ObjectInputStream(FileInputStream(file)).use { it ->
        //Read the family back from the file
        obj = it.readObject()
    }

    return obj!!
}

public fun arrTo2d(arr : DoubleArray) : Array<DoubleArray>
{
    val rowLength = sqrt(arr.size.toDouble()).toInt()
    val arr2d = Array<DoubleArray>(rowLength) {DoubleArray(rowLength)}
    var row = 0
    var col = 0
    for(i in 0 until arr.size)
    {
        if(i % sqrt(arr.size.toDouble()).toInt() == 0 && i > 0) {
            row++
            col = 0
        }
        arr2d[row][col] = arr[i]
        col++
    }
    return arr2d
}
