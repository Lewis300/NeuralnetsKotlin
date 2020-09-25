package layers

import Math.Matrix
import realAssert
import java.io.Serializable
import java.util.*

class Filter(val sideLength: Int) : Serializable {

    var filterMatrix : Matrix

    private var padding = 0
    private var stride = 1
    val rowPositionArray: Array<Int>
    val colPositionArray: Array<Int>

    init {
        filterMatrix = generateFilter(sideLength)
        rowPositionArray = makeRowPositionArray()
        colPositionArray = makeColPositionArray()
        realAssert(rowPositionArray.size == colPositionArray.size && colPositionArray.size == sideLength)
    }

    fun convolveOverMatrix(input : Matrix) : Matrix {
        realAssert(sideLength < input.height)

        val outHeight = (input.height - sideLength + 2*padding)/stride + 1
        val outWidth = (input.width - sideLength + 2*padding)/stride + 1

        val outputMatrix = Matrix(outWidth, outHeight)

        for(row in 0 until outHeight) {

            for(col in 0 until outWidth) {
                outputMatrix[row, col] = convolveOverSection(input, row, col)
            }
        }
        return outputMatrix
    }

    private fun convolveOverSection(input: Matrix, row: Int, col: Int) : Double {
        var convolution = 0.0
        for(y in 0 until sideLength)
        {
            for(x in 0 until sideLength)
            {
                convolution += filterMatrix[y, x]*(input[row+y, col+x])
            }
        }
        return convolution/(sideLength*sideLength)
    }

    private fun generateFilter(sideLength: Int) : Matrix
    {
        val filter = Matrix(sideLength, sideLength)
        val random = Random()
        for(row in 0 until sideLength)
        {
            for(col in 0 until sideLength)
            {
                filter[row, col] = random.nextDouble()
            }
        }
        //print(filter)
        return filter
    }

    operator fun get(y: Int, x: Int): Double {
        return filterMatrix[y, x]
    }

    private fun makeRowPositionArray() : Array<Int> {
        val positions = arrayListOf<Int>()
        if(sideLength%2 != 0) {
            for(i in sideLength/2 downTo -sideLength/2)
            {
                positions.add(i)
            }
        }
        else {
            for(i in sideLength/2 downTo -sideLength/2)
            {
                if(i != 0){positions.add(i)}
            }
        }
        return positions.toTypedArray()
    }

    private fun makeColPositionArray() : Array<Int> {
        val positions = arrayListOf<Int>()
        if(sideLength%2 != 0) {
            for(i in -sideLength/2 until sideLength/2 + 1)
            {
                positions.add(i)
            }
        }
        else {
            for(i in -sideLength/2 until sideLength/2 + 1)
            {
                if(i != 0){positions.add(i)}
            }
        }
        return positions.toTypedArray()
    }

    companion object {

        fun getDerivatesOfConvolutionAfterMaxPool(input: Matrix, filterSideLength: Int) : Matrix {
            val stride = filterSideLength
            val arr = Matrix(input.width, input.height)
            var row = 0
            var col = 0

            while(row < input.height)
            {
                col = 0
                while(col < input.width)
                {
                    val pair = getMaxDerivativeInScope(
                        input,
                        filterSideLength,
                        row * filterSideLength,
                        col * filterSideLength
                    )
                    arr[pair.first, pair.second] = 1.0
                    //arr[row, col] = getMaxInScope(input, filterSideLength, row*filterSideLength, col*filterSideLength)
                    col++
                }
                row++
            }
            return arr

            return arr
        }

        private fun getMaxDerivativeInScope(input: Matrix, filterSideLength: Int, row: Int, col: Int) : Pair<Int, Int>
        {
            var max = input[row, col]
            var maxRow = row
            var maxCol = col

            for(y in 0 until filterSideLength) {
                for (x in 0 until filterSideLength) {
                    var value = 0.0
                    if (row + y < input.height && col + x < input.width)
                    {
                        value = input[row + y, col + x]
                        if(value > max) {
                            max = value
                            maxRow = row+y
                            maxCol = col+x
                        }
                    }
                    else if (row + y < input.height && col + x >= input.width)
                    {
                        value = input[row + y, input.width-1]
                        if(value > max) {
                            max = value
                            maxRow = row+y
                            maxCol = input.width-1
                        }
                    }
                    else if (row + y >= input.height && col + x < input.width)
                    {
                        value = input[input.height-1, col + x]
                        if(value > max) {
                            max = value
                            maxRow = input.height-1
                            maxCol = col+x
                        }
                    }
                    else
                    {
                        value = input[input.height-1, input.width-1]
                        if(value > max) {
                            max = value
                            maxRow = input.height-1
                            maxCol = input.width-1
                        }
                    }
                }
            }
            return Pair(maxRow, maxCol)
        }
    }
}