package networks

import Math.ActivationFunction
import layers.Layer
import Math.Matrix
import layers.ConvolutionalLayer
import layers.FullyConnectedLayer
import java.io.Serializable
import java.lang.Exception
import kotlin.math.floor
import kotlin.math.sqrt

open class Network(open var layers: Array<out Layer>): Serializable {

    open lateinit var input: Matrix

    val layersArray = getLayersAsArray()

    open fun forward(input: Matrix) : Matrix
    {
        realAssert(input.height == layers[0].inputSize)

        var inputMatrix = arrayOf(input)
        this.input = input
        var lPrevious : Layer? = null
        for(i in layers.indices)
        {
            val l = layers[i]

            if(lPrevious == null){
                if(l is FullyConnectedLayer) {
                    inputMatrix = arrayOf(l.forward(inputMatrix[0]))
                }
                else if(l is ConvolutionalLayer) {
                    inputMatrix = l.forward(arrayOf(Matrix(arrTo2d(input.toDoubleArray()))))
                }
            }
            else if(lPrevious is ConvolutionalLayer) {
                if(l is FullyConnectedLayer) {
                    inputMatrix = arrayOf(l.forward(Matrix(arrayOf(flatten(inputMatrix))).transpose()))
                }
                else if(l is ConvolutionalLayer) {
                    inputMatrix = l.forward(inputMatrix)
                }
            }
            else if(lPrevious is FullyConnectedLayer){
                if(l is FullyConnectedLayer) {
                    inputMatrix = arrayOf(l.forward(inputMatrix[0]))
                }
                else if(l is ConvolutionalLayer) {
                    inputMatrix = l.forward(arrayOf(Matrix(arrTo2d(input.toDoubleArray()))))
                }
            }

            lPrevious = l
        }

        return inputMatrix[0]
    }

    open fun backpropogate(organizedGradient: Array<Layer>, target: Matrix) : Matrix {
        var dC_dA = layers[layers.size-1].getError(target)
        var dC_dA_cnn =  arrayListOf<Matrix>()

        for(i in layersArray.size-1 downTo  0) {
            val l = layersArray[i]
            if(l is FullyConnectedLayer) {
                val backpropData: FullyConnectedLayer.LayerBackpropData = if(i == 0) {
                    l.backpropogate(dC_dA, input)
                }
                else if(layersArray[i-1] is FullyConnectedLayer) {
                    l.backpropogate(dC_dA, (layersArray[i-1] as FullyConnectedLayer).a)
                }
                else {
                    val behind = (layers[i-1] as ConvolutionalLayer)
                    val x =  Matrix(arrayOf(flatten(behind.MArr.toTypedArray()))).transpose()
                    l.backpropogate(dC_dA, x)
                }

                var organizedGradientLayer = organizedGradient[i] as FullyConnectedLayer
                organizedGradientLayer.W.plusAssign(backpropData.dC_dW)
                organizedGradientLayer.b.plusAssign(backpropData.dC_db)
                dC_dA = backpropData.dC_dA
            }
            else if(l is ConvolutionalLayer) {

                if(dC_dA_cnn.isEmpty()) {
                    dC_dA_cnn = formatClassificationNetworkBackprop(dC_dA, l.OArr.size)
                }

                val list = arrayListOf<Matrix>()
                for(input in 0 until l.inputs.size) {
                    val cpy =  dC_dA_cnn.toTypedArray().copyOfRange(input*l.filters.size, (input+1)*l.filters.size)
                    list.add(l.backpropogateMultiple(organizedGradient[i] as ConvolutionalLayer, l.inputs[input], cpy))
                }
                dC_dA_cnn = list
                dC_dA = Matrix(arrayOf(flatten(dC_dA_cnn.toTypedArray()))).transpose()
            }
        }
        return dC_dA
    }

    fun dropoutAll(dropoutMap : Map<Int, IntArray>)
    {
        for(layerIndex in 0 until layers.size)
        {
            val l = layers[layerIndex]
            if(dropoutMap.containsKey(layerIndex) && l is FullyConnectedLayer)
            {
                l.dropoutNeurons(dropoutMap[layerIndex]!!)
            }
        }
    }

    fun dropInAll(dropoutMap: Map<Int, IntArray>)
    {
        for(layerIndex in 0 until layers.size)
        {
            val l = layers[layerIndex]
            if(dropoutMap.containsKey(layerIndex) && l is FullyConnectedLayer)
            {
                l.dropinNeurons(dropoutMap[layerIndex]!!)
            }
        }
    }

    fun getLayersAsArray() : Array<Layer> {
        if(layers == null){
            return arrayOf<Layer>()
        }
        else if(layers.isEmpty()){
            return arrayOf<Layer>()
        }
        else{
            return layers as Array<Layer>
        }
    }

    private fun formatClassificationNetworkBackprop(dC_dA: Matrix, numConvolutionsFlattened: Int) : ArrayList<Matrix> {
        val dC_dA_arr = dC_dA.toDoubleArray()
        val sizeOfConvolutions = dC_dA_arr.size/numConvolutionsFlattened
        val outArr = arrayListOf<Matrix>()
        for(i in 0 until numConvolutionsFlattened) {
            outArr.add(Matrix(arrTo2d(dC_dA_arr.copyOfRange(i * sizeOfConvolutions, (i + 1) * sizeOfConvolutions))))
        }
        return outArr
    }

    class Builder(val inputSize: Int, val outputSize: Int) {

        private var TYPE_FCN = "FCN"
        private var TYPE_CNN = "CNN"
        private var NETWORK_TYPE = TYPE_FCN

        private val layers = arrayListOf<Layer>()

        fun addFullyConnectedLayer(size: Int, af: ActivationFunction) : Builder {
            if(layers.isEmpty()) {
                layers.add(FullyConnectedLayer(inputSize, size, af))
            }
            else {
                layers.add(FullyConnectedLayer(getInputSizeOfNextLayer(), size, af))
            }
            return this
        }

        fun addConvolutionalLayer(filterSize: Int, numFilters: Int, maxPool: Boolean, af: ActivationFunction) : Builder {
            NETWORK_TYPE = TYPE_CNN

            if(layers.isEmpty()){
                layers.add(ConvolutionalLayer(af, filterSize, maxPool, numFilters, inputSize))
            }
            else{
                layers.add(ConvolutionalLayer(af, filterSize, maxPool, numFilters, getInputSizeOfNextLayer()))
            }
            return this
        }

        private fun getInputSizeOfNextLayer() : Int {
            if(layers.isEmpty()){return 0}

            var passThrough = arrayOf(
                when(layers[0] is ConvolutionalLayer){
                    true -> {
                        if(sqrt(inputSize + 0.0) != floor(sqrt(inputSize+0.0))) {throw Exception("Input is not a perfect square: $inputSize")}
                        Matrix(sqrt(inputSize + 0.0).toInt(), sqrt(inputSize + 0.0).toInt())
                    }
                    else -> Matrix(1, inputSize)
                }
            )
            for(layer in layers){
                if(layer is ConvolutionalLayer) {
                    passThrough = layer.forward(passThrough)
                }
                else {
                    passThrough = arrayOf(layer.forward(Matrix(arrayOf(flatten(passThrough))).transpose()))
                }
            }
            return flatten(passThrough).size
        }

        fun build(): Network {
            if(layers[layers.size-1].totalOutputSize() != outputSize)
            {
                throw NeuralNetworkBuildError("Output layer output size (${layers[layers.size-1].totalOutputSize()}) is not equal to specified output size: $outputSize")
            }
            return Network(layers.toTypedArray())
        }

        private fun flatten(matricies : Array<Matrix>) : DoubleArray {
            var outArray = arrayListOf<Double>()
            for(matrix in matricies){
                val dArr = matrix.toDoubleArray()
                for(i in 0 until dArr.size) {
                    outArray.add(dArr[i])
                }
            }
            return outArray.toDoubleArray()
        }
    }

    companion object {
        public fun flatten(matricies : Array<Matrix>) : DoubleArray {
            var outArray = arrayListOf<Double>()
            for(matrix in matricies){
                val dArr = matrix.toDoubleArray()
                for(i in 0 until dArr.size) {
                    outArray.add(dArr[i])
                }
            }
            return outArray.toDoubleArray()
        }
    }

    public fun realAssert(boolean: Boolean)
    {
        if(!boolean)
        {
            throw Exception()
        }
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

    class NeuralNetworkBuildError(message: String): Exception(message)
}