package Tools

import java.io.File
import javax.imageio.ImageIO
import kotlin.math.roundToInt

class DataLoader(val dataLocation : String) {

    private var trainingData : DataSet? = null
    private var testingData : DataSet? = null

    fun makeTargetArray(desiredNumber : Int) : DoubleArray
    {
        val array = DoubleArray(10)
        for(i in 0 until 10)
        {
            if(i == desiredNumber)
            {
                array[i] = 1.0
            }
            else
            {
                array[i] = 0.0
            }
        }
        return array
    }

    fun loadData(type: DataType, percentageTestData : Double) : DataSet.DataHolder
    {
        val dataFolder = File(dataLocation)

        var trainInputs = arrayListOf<DoubleArray>()
        var trainTargets = arrayListOf<DoubleArray>()

        var testInputs = arrayListOf<DoubleArray>()
        var testTargets = arrayListOf<DoubleArray>()

        if(type == DataType.MNIST_PNG)
        {
            val dataFilePaths = dataFolder.list()
            for(folder in dataFilePaths)
            {
                val currentFolder = File("$dataLocation/$folder").listFiles()
                for(index in 0 until currentFolder.size)
                {
                    if(index > (currentFolder.size*(1.0 - percentageTestData/100.0)).roundToInt()) {
                        testInputs.add(convertToGrayScaleArray(ImageIO.read(currentFolder[index])))
                        testTargets.add(makeTargetArray(folder.toInt()))
                    }
                    else {
                        trainInputs.add(convertToGrayScaleArray(ImageIO.read(currentFolder[index])))
                        trainTargets.add(makeTargetArray(folder.toInt()))
                    }
                    //println(Matrix(arrTo2d(convertToGrayScaleArray(ImageIO.read(currentFolder[index])))))
                }
            }
        }

        println("Train inputs size: ${trainInputs.size}")
        println("Test inputs size: ${testInputs.size}")

        trainingData = DataSet(trainInputs.toTypedArray(), trainTargets.toTypedArray())
        testingData = DataSet(testInputs.toTypedArray(), testTargets.toTypedArray())

        return DataSet.DataHolder(trainingData!!, testingData!!)
    }
}

enum class DataType {
    PNG,
    MNIST_PNG
}