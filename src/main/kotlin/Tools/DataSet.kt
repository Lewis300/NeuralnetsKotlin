package Tools

import realAssert
import java.io.Serializable

class DataSet(val inputs : Array<DoubleArray>, val targets : Array<DoubleArray>) : Serializable
{
    init {
        realAssert(inputs.size == targets.size)
    }

    fun size() : Int {return inputs.size}

    class DataHolder(val training: DataSet, val testing: DataSet) : Serializable
}