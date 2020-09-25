package layers

import Math.ActivationFunction
import Math.Matrix
import java.io.Serializable
import kotlin.math.sqrt

class ConvolutionalLayer(override var activationFunction: ActivationFunction, filterSideLength : Int, val pool: Boolean, val numFilters: Int,
                         override var inputSize: Int
    ) : Serializable, Layer() {
    override fun totalOutputSize(): Int {
        return checkOutSize(Matrix(sqrt(inputSize+0.0).toInt(), sqrt(inputSize+0.0).toInt()))
    }

    var filter : Filter
    var filters : Array<Filter>
    lateinit var input : Matrix
    lateinit var Z : Matrix
    lateinit var O : Matrix
    lateinit var M : Matrix

    lateinit var pooled : Matrix
    var pooler: MaxPoolLayer? = null

    lateinit var inputs: Array<Matrix>
    lateinit var ZArr : ArrayList<Matrix>
    lateinit var OArr : ArrayList<Matrix>
    lateinit var MArr : ArrayList<Matrix>

    init{
        if(pool){pooler = MaxPoolLayer(2)
        }

        filter = Filter(filterSideLength)
        val filterList = arrayListOf<Filter>()
        for(i in 0 until numFilters)
        {
            filterList.add(Filter(filterSideLength))
        }
        filters = filterList.toTypedArray()
    }

    override fun forward(input : Matrix) : Matrix {
        this.input = input
        val _z = filter.convolveOverMatrix(input)
        var _a = activationFunction.pass(_z)
        Z = _z
        O = _a
        //println("O height: ${O.height}")
        if(pool && pooler != null){
            M = pooler!!.forward(_a)
            return M
        }

        return O
    }

    fun forward(inputs: Array<Matrix>) : Array<Matrix> {
        this.inputs = inputs
        ZArr = arrayListOf<Matrix>()
        OArr = arrayListOf<Matrix>()
        MArr = arrayListOf<Matrix>()

        for(input in inputs){
            forwardSingleInput(input)
        }

        return when(pool) {
            true -> MArr.toTypedArray()
            false -> OArr.toTypedArray()
        }
    }

    fun forwardSingleInput(input: Matrix) {
        this.input = input

//          println("Input")
//          println(inputs[0])
        val _ZArr = arrayListOf<Matrix>()
        val _OArr = arrayListOf<Matrix>()
        val _MArr = arrayListOf<Matrix>()

        for(filterNum in 0 until filters.size) {
            val _z = filter.convolveOverMatrix(input)
            var _a = activationFunction.pass(_z)
            _ZArr.add(_z)
            _OArr.add(_a)
            //println("O height: ${O.height}")

            if(pool && pooler != null){
                val _m = pooler!!.forward(_a)
                _MArr.add(_m)
            }

        }

        ZArr.addAll(_ZArr)
        OArr.addAll(_OArr)
        MArr.addAll(_MArr)
    }

    fun backpropogateMultiple(gradientLayer: ConvolutionalLayer, input: Matrix, dC_dM_array: Array<Matrix>) : Matrix
    {
        val dC_dA_ahead = Matrix(input.width, input.height)
        for(i in 0 until dC_dM_array.size) {
            dC_dA_ahead += backpropogate(gradientLayer, input, dC_dM_array[i], filters[i], i)
        }
        //println(dC_dA_ahead)
        return dC_dA_ahead
    }

    fun backpropogate(gradientLayer: ConvolutionalLayer, input: Matrix, dC_dM: Matrix, filter: Filter, index: Int) : Matrix
    {
        // val outArray = arrayListOf<Matrix>()
        var dC_dO = dC_dM
        if(pool) {
            // println("Input height: ${input.height}")
            dC_dO = pooler!!.getDCDO(OArr[index], dC_dM)
        }
        val dO_dZ = activationFunction.derivateivePass(OArr[index])
//        println("DO_DZ")
//        println(dO_dZ)
        val dC_dF = Matrix(filter.sideLength, filter.sideLength)
        // println("dO_dZ height: ${dO_dZ.height} | dC_dO height: ${dC_dO.height}")
        val dC_dZ = dO_dZ*dC_dO
        //Get dC_dF
        for(filterRow in 0 until filter.sideLength) {
            for(filterCol in 0 until filter.sideLength) {
                dC_dF[filterRow, filterCol] = getDCDFi(dC_dZ, input, filterRow, filterCol)
            }
        }

        gradientLayer.filters[index].filterMatrix += dC_dF
//        println(dC_dF)
//        println()
        val dC_dA_ahead = Matrix(input.width, input.height)
        for(row in 0 until input.height) {
            for(col in 0 until input.width) {
                dC_dA_ahead[row, col] = getDCDXi(dC_dZ, row, col)
            }
        }
        //  println(dC_dA_ahead)
        // outArray.add(dC_dA_ahead)

        return dC_dA_ahead
    }

    fun backpropogate(gradientLayer: ConvolutionalLayer, dC_dM: Matrix) : Matrix
    {
       // val outArray = arrayListOf<Matrix>()
        var dC_dO = dC_dM
        if(pool) {
           // println("Input height: ${input.height}")
            dC_dO = pooler!!.getDCDO(O, dC_dM)
        }
        val dO_dZ = activationFunction.derivateivePass(O)
        val dC_dF = Matrix(filter.sideLength, filter.sideLength)
       // println("dO_dZ height: ${dO_dZ.height} | dC_dO height: ${dC_dO.height}")
        val dC_dZ = dO_dZ*dC_dO
        //Get dC_dF
        for(filterRow in 0 until filter.sideLength) {
            for(filterCol in 0 until filter.sideLength) {
                dC_dF[filterRow, filterCol] = getDCDFi(dC_dZ, input, filterRow, filterCol)
            }
        }

        gradientLayer.filter.filterMatrix += dC_dF
//        println(dC_dF)
//        println()
        val dC_dA_ahead = Matrix(input.width, input.height)
        for(row in 0 until input.height) {
            for(col in 0 until input.width) {
                dC_dA_ahead[row, col] = getDCDXi(dC_dZ, row, col)
            }
        }
        //  println(dC_dA_ahead)
       // outArray.add(dC_dA_ahead)
   
        return dC_dA_ahead
    }

    private fun getDCDXi(dC_dO: Matrix, inputRow: Int, inputCol: Int) : Double {
        var dC_dXi = 0.0
        // sum over k
            //dC_dOk * dOk_dXi
        for(row in 0 until dC_dO.height) {
            for(col in 0 until dC_dO.width) {
                //println(getDOkDXi(row, col, inputRow, inputCol))
                dC_dXi += dC_dO[row, col]*getDOkDXi(row, col, inputRow, inputCol)
            }
        }
        return dC_dXi
    }
    //                                     k                             i
    //                             r               c               n           p
    private fun getDOkDXi(outputRow: Int, outputCol: Int, inputRow: Int, inputCol: Int) : Double {
        val denominator = filter.sideLength*filter.sideLength


        val filterRow = (inputRow-outputRow)
        val filterCol = (inputCol-outputCol)

        if(filterRow !in 0 until filter.sideLength || filterCol !in 0 until filter.sideLength){
          //  println(0.0)
            return 0.0
        }
        //println(filter.filterMatrix[filterRow, filterCol]/denominator)
//        //println("IndexOfCol: ${inputCol-outputCol}")
        return filter.filterMatrix[filterRow, filterCol]/denominator
    }

    //                                                  k                           i
    //                          X           r                   c           j               k
    private fun getDOkDFi(input: Matrix, outputRow: Int, outputCol: Int, filterRow: Int, filterCol: Int) : Double {

        val inputRow = outputRow + filterRow
        val inputCol = outputCol + filterCol

        val denominator = filter.sideLength*filter.sideLength
        return input[inputRow, inputCol]/denominator
    }
    //                                                i
    private fun getDCDFi(dC_dO: Matrix, input: Matrix, filterRow: Int, filterCol: Int) : Double {
        var dC_df = 0.0
        for(row in 0 until dC_dO.height)
        {
            for(col in 0 until dC_dO.width) {
                val add = dC_dO[row, col]*getDOkDFi(input, row, col, filterRow, filterCol)
                //if(add != 0.0){println(add)}
                dC_df += add
            }
        }
        return dC_df
    }

    override fun getError(target: Matrix): Matrix {
        return Matrix(arrayOf(flatten(OArr.toTypedArray()))) -target
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
        //return matrix.toDoubleArray()
    }

    fun checkOutSize(input: Matrix) : Int {

        var passThrough = forward(arrayOf(input))

        val toClassification = flatten(passThrough)
        return toClassification.size
    }

    companion object {
        fun getOutputSize(inputSideLength: Int, filterSideLength: Int, padding: Int, stride: Int) : Int
        {
            return (inputSideLength - filterSideLength + 2*padding)/stride + 1
        }
    }
}