import Math.Matrix
import Tools.DataSet
import layers.*
import networks.Network
import kotlin.math.pow
import kotlin.random.Random

class Trainer
{
    private lateinit var network: Network
    private lateinit var organizedGradient: Array<Layer>
    private var learningRate: Double = 0.001
    private var batchSize: Int = 100
    private var epochs: Int = 1000

    fun trainNetwork(network: Network, dataSize: Int, batchSize: Int, epochs: Int, lr: Double, dropoutChance : Double, dataSet: DataSet)
    {
        realAssert(dataSet.inputs.size == dataSet.targets.size)
        realAssert(dataSet.inputs.size == dataSize)

        this.network = network
        this.batchSize = batchSize
        this.epochs = epochs
        this.learningRate = lr
        this.organizedGradient = buildOrganizedGradient(network) // Will hold the sum of cost derivatives for each weight/bias of the network (values initialized to 0)

        val batches = dataSize/batchSize
        for(epoch in 0 until epochs)
        {
            var epochCost = 0.0
            for(batch in 0..batches)
            {
                val batchPairing = randomSampleInputTargetPairs(dataSet, batchSize)
                val batchInputs = batchPairing[0]
                val batchTargets = batchPairing[1]
                var batchCost = 0.0
                var sampleCost: Double

                val dropoutMap = getDropoutMap(dropoutChance)

                network.dropoutAll(dropoutMap)

                for(index in 0 until batchSize)
                {
                    sampleCost = runSample(batchInputs[index], batchTargets[index])
                    batchCost += sampleCost
                    epochCost += sampleCost
                }
                onBatchFinished()
                network.dropInAll(dropoutMap)
                if(batch % (batches/10) == 0)
                {
                    println("Finished batch, epoch: $batch, $epoch | Cost: ${batchCost/batchSize}")
                }
            }

            if(epoch % 1 == 0)
            {
                println("Finished epoch: $epoch | Cost: ${epochCost/dataSize}")
            }
        }
    }

    private fun runSample(input: DoubleArray, target: DoubleArray) : Double
    {
        val output = network.forward(Matrix(arrayOf(input)).transpose())
        val targetMatrix = Matrix.arrayToMatrix(target)

        network.backpropogate(organizedGradient, targetMatrix)

        var sum = 0.0
        val outputArr = output.toDoubleArray()
        for(i in 0 until outputArr.size)
        {
            sum += (outputArr[i] - target[i]).pow(2)
        }

        return sum
    }

    private fun onBatchFinished()
    {
        var gradientLayer : Layer
        for(i in organizedGradient.indices)
        {
            gradientLayer = organizedGradient[i]

            if(gradientLayer is FullyConnectedLayer)
            {
                gradientLayer.W = gradientLayer.W.applyFunctionToAllValues { x: Double -> convertToDelta(x) }
            }
            else if(gradientLayer is ConvolutionalLayer)
            {
                //gradientLayer.filter.filterMatrix = gradientLayer.filter.filterMatrix.applyFunctionToAllValues { x: Double -> convertToDelta(x) }
                for(filter in gradientLayer.filters) {
                    filter.filterMatrix = filter.filterMatrix.applyFunctionToAllValues { x: Double -> convertToDelta(x)  }
                }
            }
        }

        for(layer in 0 until network.layers.size)
        {
            val l = network.layers[layer]
            if(l is FullyConnectedLayer)
            {
                val ol = organizedGradient[layer] as FullyConnectedLayer
                val l = network.layers[layer] as FullyConnectedLayer
                l.W.minusAssign(ol.W)
                l.b.minusAssign(ol.b)
            }
            else if(l is ConvolutionalLayer) {
                val organizedLayer = organizedGradient[layer] as ConvolutionalLayer
                for(i in 0 until l.filters.size) {
                    l.filters[i].filterMatrix.minusAssign(organizedLayer.filters[i].filterMatrix)
                }
            }
        }
        organizedGradient = buildOrganizedGradient(network)
    }

    private fun buildOrganizedGradient(network: Network) : Array<Layer>
    {
        var layers = network.getLayersAsArray()

        val gradientBuilder = arrayListOf<Layer>()
        for(l in layers) {
            if (l is FullyConnectedLayer) {
                realAssert(l.b.width == 1)
                val W = Matrix(l.W.width, l.W.height)
                val b = Matrix(l.b.width, l.b.height)

                W.fillZeros()
                b.fillZeros()
                val fcl = FullyConnectedLayer(W.height, W.width, l.activationFunction)
                fcl.W = W
                fcl.b = b
                gradientBuilder.add(fcl)
            } else if (l is ConvolutionalLayer)
            {
                val filters = arrayListOf<Filter>()//layers.Filter(l.filters[0].sideLength)
                val cl = ConvolutionalLayer(
                    l.activationFunction,
                    l.filter.sideLength,
                    l.pool,
                    l.numFilters,
                    l.inputSize
                )

                for(f in 0 until l.filters.size) {
                    val fil = Filter(l.filters[f].sideLength)
                    fil.filterMatrix.fillZeros()
                    filters.add(fil)
                }
                cl.filters = filters.toTypedArray()

                gradientBuilder.add(cl)
            }
        }

        return gradientBuilder.toTypedArray()
    }

    private fun randomSampleInputTargetPairs(dataSet: DataSet, sampleSize: Int) : Array<Array<DoubleArray>>
    {
        val rand = Random
        val newInputs = arrayListOf<DoubleArray>()
        val newTargets = arrayListOf<DoubleArray>()
        for(i in 0 until sampleSize)
        {
            val index = rand.nextInt(0, dataSet.size())
            newInputs.add(dataSet.inputs[index])
            newTargets.add(dataSet.targets[index])
        }
        return arrayOf(newInputs.toTypedArray(), newTargets.toTypedArray())
    }

    private fun getDropoutMap(chancePerNeuron : Double) : Map<Int, IntArray>
    {
        val map = mutableMapOf<Int, IntArray>()
        for(layer in 0 until network.layers.size-1)
        {
            val arr = arrayListOf<Int>()
            if(network.layersArray[layer] is FullyConnectedLayer)
            {
                for(neuron in 0 until network.layers[layer].totalOutputSize())
                {
                    if(Random.nextDouble() <= chancePerNeuron/100.0)
                    {
                        arr.add(neuron)
                    }
                }
            }
            map[layer] = arr.toIntArray()
        }
        return map
    }

    fun convertToDelta(x: Double) : Double {
        return learningRate*(x/batchSize.toDouble())
    }
}