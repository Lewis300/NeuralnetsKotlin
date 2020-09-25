package layers

import Math.ActivationFunction
import Math.Matrix
import java.io.Serializable

abstract class Layer : Serializable {
    abstract fun totalOutputSize() : Int
    abstract var activationFunction : ActivationFunction
    abstract fun forward(input : Matrix) : Matrix

    abstract fun getError(target: Matrix) : Matrix
    abstract var inputSize: Int
}