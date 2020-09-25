package Math

import java.io.Serializable
import java.lang.Exception
import Math.Matrix

public fun realAssert(boolean: Boolean)
{
    if(!boolean)
    {
        throw Exception()
    }
}

interface ActivationFunction : Serializable
{
    fun pass(input : Matrix) : Matrix
    fun derivateivePass(input : Matrix) : Matrix
    fun inverse(input : Matrix) : Matrix

    object Identity : ActivationFunction {
        override fun pass(input: Matrix): Matrix {
            return input
        }

        override fun derivateivePass(input: Matrix): Matrix {
            return input
        }

        override fun inverse(input: Matrix): Matrix {
           return input
        }

    }

    object Sigmoid : ActivationFunction {
        override fun pass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> pass(x) }
        }

        private fun pass(num: Double): Double {
            return 1.0/(1+ Math.exp(-num))
        }

        override fun derivateivePass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> derivatiavePass(x) }
        }

        override fun inverse(input: Matrix): Matrix {
            TODO("Not yet implemented")
        }

        private fun derivatiavePass(num: Double): Double {
            val sig = pass(num)
            return sig*(1-sig)
        }
    }

    object ReLU : ActivationFunction {

        override fun pass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> pass(x) }
        }

        private fun pass(num: Double): Double {
            return when(num>0)
            {
                true -> num
                false -> 0.0
            }
        }

        override fun derivateivePass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> derivatiavePass(x) }
        }

        private fun derivatiavePass(num: Double): Double {
            return when(num>0)
            {
                true -> 1.0
                false -> 0.0
            }
        }
        override fun inverse(input: Matrix): Matrix {
            TODO("Not yet implemented")
        }

    }

    object Softmax: ActivationFunction {

        override fun pass(input: Matrix): Matrix {
            var sum: Double = 0.0
            val inputAsArray = input.toDoubleArray()
            for (num in inputAsArray) {
                sum += Math.exp(num)
            }

            val softmaxArray = arrayListOf<Double>()
            for (num in inputAsArray) {
                softmaxArray.add(Math.exp(num) / sum)
            }
            return Matrix.arrayToMatrix(softmaxArray.toDoubleArray())
        }

        override fun derivateivePass(input: Matrix): Matrix {
            val result = Matrix(input.height, input.height)
            val aAsArray = input.toDoubleArray()

            for(row in 0 until input.height)
            {
                for(col in 0 until input.height)
                {
                    if(row == col) {result[row, col] =
                        aAsArray[row]*(1 - aAsArray[col])}
                    else{result[row, col] = -aAsArray[row]*aAsArray[col]}
                }
            }

            return result
        }

        override fun inverse(input: Matrix): Matrix {
            return pass(input)
        }

    }

    object Step : ActivationFunction {
        override fun pass(input: Matrix): Matrix
        {
            return input.applyFunctionToAllValues { x -> (pass(x)) }
        }

        private fun pass (num : Double) : Double {
            return when (num>=0.0){
                true -> 1.0
                false -> 0.0
            }
        }

        override fun derivateivePass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> 0.0 }
        }

        override fun inverse(input: Matrix): Matrix {
            TODO("Not yet implemented")
        }

    }

    object LeakyReLU : ActivationFunction {
        override fun pass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> pass(x) }
        }

        private fun pass(num: Double): Double {
            return when(num>0)
            {
                true -> num
                false -> 0.05*num
            }
        }

        override fun derivateivePass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> derivatiavePass(x) }
        }

        private fun derivatiavePass(num: Double): Double {
            return when(num>0)
            {
                true -> 1.0
                false -> 0.05
            }
        }

        override fun inverse(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> inverse(x) }
        }

        private fun inverse(num: Double): Double {
            return if(num>=0){
                num
            } else {
                num/0.05
            }
        }
    }

    object ReLU6 : ActivationFunction {
        override fun pass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> pass(x) }
        }

        private fun pass(num: Double): Double {
            if(num<0){return 0.0}
            else if (num >= 6.0) {return 6.0}
            else{return num}
        }

        override fun derivateivePass(input: Matrix): Matrix {
            return input.applyFunctionToAllValues { x -> derivatiavePass(x) }
        }

        private fun derivatiavePass(num: Double): Double {
            if(0.0 <num && num < 6.0){return 1.0}
            return 0.0
        }

        override fun inverse(input: Matrix): Matrix {
            TODO("Not yet implemented")
        }

    }
}
