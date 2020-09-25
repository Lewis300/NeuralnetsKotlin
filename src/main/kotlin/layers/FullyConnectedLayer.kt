package layers

import Math.ActivationFunction
import Math.Matrix
import realAssert
import java.io.Serializable
import kotlin.math.sqrt
import java.util.Random

class FullyConnectedLayer(override var inputSize : Int, val outputSize : Int, override var activationFunction: ActivationFunction) : Layer(), Serializable
{
    constructor(previousLayer : Layer, outputSize: Int, activationFunction: ActivationFunction) : this(previousLayer.totalOutputSize() , outputSize, activationFunction)
    var W = Matrix(inputSize, outputSize)
    lateinit var z: Matrix
    lateinit var a: Matrix
    var b = Matrix(1, outputSize)
    lateinit var input : Matrix

    var isInputLayer: Boolean? = null

    init{
        heInitialization(W, inputSize)
        b.fillZeros()
    }

    override fun forward(input : Matrix) : Matrix
    {
        val bool = W.width == inputSize && W.width == input.height
        if(!bool) {
            println("input height : ${input.height}")
            println("N0 : ${W.width}")
        }
        realAssert(bool)
        this.input = input

        val W_dot_a0 = Matrix.dot(W, input)
        z = Matrix.add(W_dot_a0, b)
        a = activationFunction.pass(z)
        if(droppedNers.isNotEmpty())
        {
            for(neuron in droppedNers)
            {
                a[neuron, 0] = 0.0
                z[neuron, 0] = 0.0
            }
        }

        realAssert(z.height == a.height)
        realAssert(a.height == a.height)

        return a
    }

    fun backpropogate(dC_dA: Matrix, behindActivations: Matrix): LayerBackpropData {
        val dA_dZ = activationFunction.derivateivePass(a)
        val dC_dZ = getDCDZ(dC_dA, dA_dZ)

        val dC_dW = Matrix.dot(dC_dZ, behindActivations.transpose())
        val dC_dA_behind = Matrix.dot(W.transpose(), dC_dZ)

        return LayerBackpropData(dC_dW, dC_dZ, dC_dA_behind)
    }

    fun runInputBackwards(input: Matrix, behind: FullyConnectedLayer) : Matrix
    {
        val WTransposed = W.transpose()

        val W_dot_a0 = Matrix.dot(WTransposed, input)
        z = Matrix.add(W_dot_a0, behind.b)
        a = behind.activationFunction.inverse(z)
        return a
    }

    fun runInputBackwardsAsFinalLayer(input: Matrix) : Matrix
    {
        val WTransposed = W.transpose()

        val W_dot_a0 = Matrix.dot(WTransposed, input)
        z = W_dot_a0
        a = z
        return a
    }

    override fun toString(): String {
        return "Weights\n"+W.toString()+"Biases\n"+b.toString()
    }

    fun heInitialization(W : Matrix, behindLayerSize : Int)
    {
        val random = Random()
        for(row in 0 until W.height)
        {
            for(col in 0 until W.width)
            {
                W[row, col] = random.nextGaussian()*sqrt(2.0/behindLayerSize)
            }
        }
    }

    var droppedNers : ArrayList<Int> = arrayListOf<Int>()
    fun dropoutNeurons(neurons : IntArray)
    {
        for(neuron in neurons)
        {
            droppedNers.add(neuron)
        }
    }

    fun dropinNeurons(neurons: IntArray)
    {
        for(neuron in neurons)
        {
            if(droppedNers.contains(neuron))
            {
                droppedNers.remove(neuron)
            }
        }
    }

    override fun totalOutputSize(): Int {
        return outputSize
    }

    private fun getDCDZ(dC_dA : Matrix, dA_dZ : Matrix) : Matrix {

        var dC_dZ = if (activationFunction is ActivationFunction.Softmax) {
            Matrix.dot(dA_dZ, dC_dA)
        } else {
            dC_dA * dA_dZ
        }
        return dC_dZ

    }

    override fun getError(target: Matrix): Matrix {
        return a-target
    }

    data class LayerBackpropData(val dC_dW: Matrix, val dC_db: Matrix, val dC_dA: Matrix){}
}
