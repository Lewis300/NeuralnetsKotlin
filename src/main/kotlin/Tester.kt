import Math.Matrix
import Tools.DataSet
import networks.Network

class Tester(val network: Network, val dataSet: DataSet, val accuracyFunction: (DoubleArray, DoubleArray) -> Double)
{
    fun test() : Double
    {
        var input : Matrix
        var output : DoubleArray = doubleArrayOf()
        var avgAccuracy : Double = 0.0

        for(i in 0 until dataSet.size())
        {
            input = Matrix(arrayOf(dataSet.inputs[i])).transpose()
            output = network.forward(input).toDoubleArray()
            val acc = accuracyFunction(output, dataSet.targets[i])
            avgAccuracy += acc
        }

        return avgAccuracy/dataSet.size()
    }
}