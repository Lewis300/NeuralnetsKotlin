package layers

import Math.Matrix
import java.io.Serializable

class MaxPoolLayer(val poolFilterSideLength: Int) : Serializable{

    fun forward(input : Matrix) : Matrix
    {
        return maxpool(input, poolFilterSideLength)
    }

    fun getDCDO(input : Matrix, dC_dM: Matrix) : Matrix {
        return getDerivative(
            input,
            dC_dM,
            poolFilterSideLength
        ) // dC_dO
    }

    companion object {
        fun maxpool(input: Matrix, filterSideLength: Int) : Matrix {
            val stride = filterSideLength

            val outHeight = if (input.height % filterSideLength == 0) input.height/filterSideLength else input.height/filterSideLength + 1
            val outWidth = if (input.width % filterSideLength == 0) input.width/filterSideLength else input.width/filterSideLength + 1

            val arr = Matrix(outWidth, outHeight)

            var row = 0
            var col = 0

            while(row < outHeight)
            {
                col = 0
                while(col < outWidth)
                {
                    arr[row, col] = getMaxInScope(
                        input,
                        filterSideLength,
                        row * filterSideLength,
                        col * filterSideLength
                    )
                    col++
                }
                row++
            }
            return arr
        }

        fun getDerivative(input: Matrix, dC_dM: Matrix, filterSideLength: Int) : Matrix {
            val stride = filterSideLength

            val arr = Matrix(input.width, input.height)

            var row = 0
            var col = 0

            while(row < arr.height-filterSideLength)
            {
                col = 0
                while(col < arr.width - filterSideLength)
                {
                    for(r in 0 until filterSideLength) {
                        for(c in 0 until filterSideLength) { // dM_dO
                            arr[row + r, col + c] = isMaxInScope(
                                input,
                                filterSideLength,
                                row,
                                col,
                                r,
                                c
                            ) *dC_dM[row/stride, col/stride]
                        }
                    }
                    col += stride
                }
                row += stride
            }
            return arr
        }

        private fun isMaxInScope(input: Matrix, filterSideLength: Int, row: Int, col: Int, filterRow: Int, filterCol: Int) : Double {
            val value = input[row+filterRow, col+filterCol]

            for(y in 0 until filterSideLength) {
                for (x in 0 until filterSideLength) {

                   if (row + y < input.height && col + x < input.width)
                   {
                       if(value < input[row + y, col + x]){
                           return 0.0
                       }
                   }
                   else if (row + y < input.height && col + x >= input.width)
                   {
                       if(value < input[row + y, input.width-1]){
                           return 0.0
                       }
                   }
                   else if (row + y >= input.height && col + x < input.width)
                   {
                       if(value < input[input.height-1, col + x]){
                           return 0.0
                       }
                   }
                   else
                   {
                       if(value < input[input.height-1, input.width-1]) {
                           return 0.0
                       }
                   }
                }
            }
            return 1.0
        }

        private fun getMaxInScope(input: Matrix, filterSideLength: Int, row: Int, col: Int) : Double
        {
            val arr = arrayListOf<Double>()

            for(y in 0 until filterSideLength) {
                for (x in 0 until filterSideLength) {
                    var value = 0.0
                    if (row + y < input.height && col + x < input.width)
                    {
                        value = input[row + y, col + x]
                    }
                    else if (row + y < input.height && col + x >= input.width)
                    {
                        value = input[row + y, input.width-1]
                    }
                    else if (row + y >= input.height && col + x < input.width)
                    {
                        value = input[input.height-1, col + x]
                    }
                    else
                    {
                        value = input[input.height-1, input.width-1]
                    }
                    arr.add(value)
                }
            }
            return if(arr.size == 0) {
                0.0
            } else {
                arr.max()!!
            }
        }

    }
}