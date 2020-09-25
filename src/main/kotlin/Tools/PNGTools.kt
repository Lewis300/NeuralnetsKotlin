package Tools

import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.IOException


@Throws(IOException::class)
fun convertTo2DDoubleArray(inputImage: BufferedImage): Array<DoubleArray> {

    val pixels = (inputImage.raster
        .dataBuffer as DataBufferByte).data // get pixel value as single array from buffered Image
    val width = inputImage.width //get image width value
    val height = inputImage.height //get image height value
    val result =
        Array(height) { DoubleArray(width) } //Initialize the array with height and width

    //this loop allocates pixels value to two dimensional array
    var pixel = 0
    var row = 0
    var col = 0
    while (pixel < pixels.size) {
        var argb = 0
        argb = pixels[pixel].toInt()
        if (argb < 0) { //if pixel value is negative, change to positive //still weird to me
            argb += 256
        }
        result[row][col] = argb/255.0
        col++
        if (col == width) {
            col = 0
            row++
        }
        pixel++
    }
    return result
}

fun convertTo2dGrayScaleArray(inputImage: BufferedImage) {
    val arr = Array(28) { IntArray(28) }

    for (i in 0..27) {
        for (j in 0..27){

            arr[i][j] = inputImage.getRGB(i, j)
           // print(arr[i][j])
        }
        println()
    }
}

fun convertToGrayScaleArray(inputImage: BufferedImage) : DoubleArray
{
    val pixels = (inputImage.raster
        .dataBuffer as DataBufferByte).data // get pixel value as single array from buffered Image
    val width = inputImage.width //get image width value
    val height = inputImage.height //get image height value
    val result = DoubleArray(width*height)  //Initialize the array with height and width

    //this loop allocates pixels value to two dimensional array
    var pixel = 0
    var row = 0
    var col = 0
    while (pixel < pixels.size) {
        var argb = 0
        argb = pixels[pixel].toInt()
        if (argb < 0) { //if pixel value is negative, change to positive //still weird to me
            argb += 256
        }
        result[row*width + col] = argb/255.0
        col++
        if (col == width) {
            col = 0
            row++
        }
        pixel++
    }
    return result
}
