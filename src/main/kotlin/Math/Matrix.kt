package Math

import CudaFunctions
import java.io.Serializable
import java.util.*


class Matrix(val width: Int, val height: Int) : Serializable
{
    private var arr: Array<DoubleArray> = makeArr(width, height)

    fun isColumnVector() : Boolean {return arr.size == 1 && arr[0].isNotEmpty() }

    fun isRowVector() : Boolean {return arr.size > 0 && arr[0].size == 1}

    private fun makeArr(width: Int, height: Int) : Array<DoubleArray>
    {
        var newArr : ArrayList<DoubleArray> = arrayListOf()
        for(i in 0..height-1)
        {
            newArr.add(DoubleArray(width))
        }

        return newArr.toTypedArray()
    }

    constructor(arr: Array<DoubleArray>) : this(arr[0].size, arr.size)
    {
        this.arr = arr
    }

    fun setAllValsRandom(min: Double, max : Double)
    {
        realAssert(max > min)
        val rand = kotlin.random.Random
        arr.forEach {
            for(i in 0..it.size-1)
            {
                it[i] = rand.nextDouble(min, max)
            }
        }
    }

    fun setAllValsRandom()
    {
        arr.forEach {
            for(i in 0..it.size-1)
            {
                it[i] = Random().nextGaussian()
            }
        }
    }

    override fun toString(): String
    {
        var str : String = ""
        arr.forEach{
            it.forEach { str+=("$it, ") }
            str+="\b\b\n"
        }
        return str
    }

    fun applyFunctionToAllValues(func : (Double) -> Double) : Matrix
    {
        val newMatrix = Matrix(width, height)
        for(i in 0..arr.size-1)
        {
            for(j in 0..arr[0].size-1)
            {
                newMatrix.arr[i][j] = func(arr[i][j])
            }
        }
        return newMatrix
    }

    fun fillZeros()
    {
        arr.forEach {
            for(i in 0..it.size-1)
            {
                it[i] = 0.0
            }
        }
    }

    fun fill(value: Double)
    {
        arr.forEach {
            for(i in 0..it.size-1)
            {
                it[i] = value
            }
        }
    }

    /**
     * precondition: Matrix is column or row vector
     */
    fun toDoubleArray() : DoubleArray
    {
        //realAssert(isColumnVector() || isRowVector())

        val newArr = arrayListOf<Double>()
        for(i in 0..arr.size-1)
        {
            for(j in 0..arr[0].size-1)
            {
                newArr.add(arr[i][j])
            }
        }
        return newArr.toDoubleArray()
    }


    fun toFloatArray() : FloatArray {
        val newArr = arrayListOf<Float>()
        for(i in 0..arr.size-1)
        {
            for(j in 0..arr[0].size-1)
            {
                newArr.add(arr[i][j].toFloat())
            }
        }
        return newArr.toFloatArray()
    }

    fun toColumnMajor() : FloatArray {
        val newArr = arrayListOf<Float>()
        for(col in 0..arr[0].size-1)
        {
            for(row in 0..arr.size-1)
            {
                newArr.add(arr[row][col].toFloat())
            }
        }
        return newArr.toFloatArray()
    }

    operator fun get(row : Int, col: Int) : Double
    {
//        println("Row $row")
//        println("Col $col")
        return arr[row][col]
    }

    operator fun times(other: Matrix) : Matrix
    {
        realAssert(width == other.width)
        realAssert(height == other.height)

        val newMatrix = Matrix(width, height)
        for(i in 0..arr.size-1)
        {
            for(j in 0..arr[0].size-1)
            {
                newMatrix.arr[i][j] = arr[i][j]*other.arr[i][j]
            }
        }
        return newMatrix
    }

    fun transpose() : Matrix
    {
        val transpose = Array(width) { DoubleArray(height) }
        for (i in 0..height - 1) {
            for (j in 0..width - 1) {
                transpose[j][i] = arr[i][j]
            }
        }

        return Matrix(transpose)
    }

    operator fun set(row: Int, col: Int, value: Double) {
        arr[row][col] = value
    }

    operator fun minus(other: Matrix) : Matrix {
        realAssert(arr.size > 0)
        realAssert(other.arr.size > 0)
        realAssert(arr.size == other.arr.size)
        realAssert(arr[0].size > 0)
        realAssert(other.arr[0].size > 0)
        realAssert(arr[0].size == other.arr[0].size)

        realAssert(width == other.width)
        realAssert(height == other.height)

        val summand = Matrix(width, height)

        for(i in 0 until height)
        {
            for(j in 0 until width)
            {
                summand.arr[i][j] = arr[i][j] - other.arr[i][j]
            }
        }
        return summand
    }

    operator fun minusAssign(other: Matrix)
    {
        realAssert(arr.size > 0)
        realAssert(other.arr.size > 0)
        realAssert(arr.size == other.arr.size)
        realAssert(arr[0].size > 0)
        realAssert(other.arr[0].size > 0)
        realAssert(arr[0].size == other.arr[0].size)

        realAssert(width == other.width)
        realAssert(height == other.height)


        for(i in 0 until height)
        {
            for(j in 0 until width)
            {
                arr[i][j] -= other.arr[i][j]
            }
        }
    }

    operator fun plusAssign(other: Matrix)  {
        realAssert(arr.size > 0)
        realAssert(other.arr.size > 0)
        realAssert(arr.size == other.arr.size)
        realAssert(arr[0].size > 0)
        realAssert(other.arr[0].size > 0)
        realAssert(arr[0].size == other.arr[0].size)

        realAssert(width == other.width)
        realAssert(height == other.height)


        for(i in 0 until height)
        {
            for(j in 0 until width)
            {
                arr[i][j] += other.arr[i][j]
            }
        }
    }

    companion object
    {
        public fun dot(a : Matrix, b : Matrix): Matrix {
            val r1 = a.height
            val c1 = a.width
            val c2 = b.width

            val product = Array(r1) { DoubleArray(c2) }
            for (i in 0..r1 - 1) {
                for (j in 0..c2 - 1) {
                    for (k in 0..c1 - 1) {
                        product[i][j] += a.arr[i][k] * b.arr[k][j]
                    }
                }
            }

            val matrix = Matrix(product)
            return matrix
           // val cf = CudaFunctions()

           // return CudaFunctions.matrixDot(a, b)
        }

        public fun add(a : Matrix, b : Matrix) : Matrix
        {
            realAssert(a.arr.size > 0)
            realAssert(b.arr.size > 0)
            realAssert(a.arr.size == b.arr.size)
            realAssert(a.arr[0].size > 0)
            realAssert(b.arr[0].size > 0)
            realAssert(a.arr[0].size == b.arr[0].size)

            realAssert(a.width == b.width)
            realAssert(a.height == b.height)

            val summand = Matrix(a.width, a.height)

            for(i in 0 until a.height)
            {
                for(j in 0 until a.width)
                {
                    summand.arr[i][j] = a.arr[i][j] + b.arr[i][j]
                }
            }
            return summand
        }

        public fun arrayToMatrix(arr: Array<DoubleArray>) : Matrix
        {
            return Matrix(arr)
        }

        public fun arrayToMatrix(arr: Array<Double>) : Matrix
        {
            var newArr = arrayListOf<DoubleArray>()
            for(d in arr)
            {
                newArr.add(doubleArrayOf(d))
            }

            return Matrix(newArr.toTypedArray())
        }

        public fun arrayToMatrix(arr: DoubleArray) : Matrix
        {
            var newArr = arrayListOf<DoubleArray>()
            for(d in arr)
            {
                newArr.add(doubleArrayOf(d))
            }

            return Matrix(newArr.toTypedArray())
        }

        public fun realAssert(boolean: Boolean)
        {
            if(!boolean)
            {
                throw Exception()
            }
        }
    }

    fun sum() : Double {
        var sum = 0.0
        for(row in 0 until arr.size)
        {
            for(col in 0 until arr[row].size)
            {
                sum += arr[row][col]
            }
        }

        return sum
    }

    fun removeRows(rows: Set<Int>) : Matrix {

        val newArr = arrayListOf<DoubleArray>()
        for(row in 0 until arr.size) {
            if(row !in rows)
            {
                newArr.add(arr[row])
            }
        }

        return Matrix(newArr.toTypedArray())
    }

    fun removeRow(rowToRemove : Int) : Matrix {
        val newArr = arrayListOf<DoubleArray>()
        for(row in 0 until arr.size) {
            if(row != rowToRemove)
            {
                newArr.add(arr[row])
            }
        }
        return Matrix(newArr.toTypedArray())
    }

    fun addRows(rowMap : Map<Int, DoubleArray>) : Matrix {
        val arrList = arr.toMutableList()
        for(row in 0 until arr.size) {
            if(row in rowMap.keys)
            {
                arrList.add(row, rowMap[row]!!)
            }
        }

        return Matrix(arrList.toTypedArray())
    }

    fun addRow(index : Int, data : DoubleArray) : Matrix {
        val arrList = arr.toMutableList()
        for(row in 0 until arr.size) {
            if(row == index)
            {
                arrList.add(row, data)
            }
        }

        return Matrix(arrList.toTypedArray())
    }

    fun getRow(row : Int) : DoubleArray {
        return arr[row]
    }

    fun getRowAsFloat(row: Int) : FloatArray
    {
        val floatArray = FloatArray(arr[row].size)
        for (i in 0 until arr[row].size) {
            floatArray[i] = arr[row][i].toFloat()
        }
        return floatArray
    }

    fun removeColumns(cols: Set<Int>) : Matrix {

        var newArr = transpose()
        newArr = newArr.removeRows(cols)
        return newArr.transpose()
    }

    fun removeColumn(rowToRemove : Int) : Matrix {
        var newArr = transpose()
        newArr = newArr.removeRow(rowToRemove)
        return newArr.transpose()
    }

    fun addColumns(colMap : Map<Int, DoubleArray>) : Matrix {
        var newArr = transpose()
        newArr = newArr.addRows(colMap)
        return newArr.transpose()
    }

    fun addColumn(index : Int, data : DoubleArray) : Matrix {
        val newArr = transpose()
        newArr.addRow(index, data)
        return newArr.transpose()
    }

    fun getColumn(col : Int) : DoubleArray {
        return transpose().getRow(col)
    }

    fun getColumnAsFloat(col : Int) : FloatArray {
        val fArr = FloatArray(arr.size)
        for(row in 0 until arr.size) {
            fArr[row] = arr[row][col].toFloat()
        }
        return fArr
    }

    fun toArray() : Array<DoubleArray> {
        return arr.clone()
    }

    fun setRow(row : Int, rowList: DoubleArray)
    {
        arr[row] = rowList
    }

    fun setCol(col : Int, colList: DoubleArray)
    {
        for(i in 0 until height)
        {
            arr[i][col] = colList[i]
        }
    }

    fun setRowZeros(row : Int)
    {
        for(i in 0 until width)
        {
            arr[row][i] = 0.0
        }
    }

    fun setColZeros(col : Int)
    {
        for(i in 0 until height)
        {
            arr[i][col] = 0.0
        }
    }
}