import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import Math.Matrix;
import jcuda.jcublas.cublasPointerMode;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import java.util.ArrayList;
import java.util.Arrays;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.*;

public class CudaFunctions {

    public static cublasHandle handle = new cublasHandle();

    static Pointer d_A = new Pointer();
    static Pointer d_B = new Pointer();
    static Pointer d_C = new Pointer();

    static float h_A[];
    static float h_B[];
    static float h_C[];

    static float alpha = 1.0f;
    static float beta = 0.0f;

    public static void main(String[] args) {

        Matrix a = new Matrix(1, 2);
        a.setAllValsRandom();

        Matrix b = new Matrix(1, 2);
        b.setAllValsRandom();

        Matrix m = matrixDot(a, b);
    }

    public static Matrix matrixDot(Matrix A, Matrix B) {

//        double[] aRow, bCol;
//        double vectorDot;
//
//        Pointer deviceDataA = new Pointer();
//        Pointer deviceDataB = new Pointer();
//
//        cudaFree(deviceDataA);
//
//        Matrix outMatrix = new Matrix(b.getWidth(), a.getHeight());
//
//        for(int col = 0; col<b.getWidth(); col++) {
//            for(int row = 0; row<a.getHeight(); row++) {
//                aRow = a.getRow(row);
//                bCol = b.getColumn(col);
//
//                vectorDot = vectorDot(aRow, bCol, deviceDataA, deviceDataB, handle);
//                outMatrix.set(row, col, vectorDot);
//            }
//        }
//
//        return outMatrix;




        h_A = A.toColumnMajor();
        h_B = B.toColumnMajor();
        h_C = new float[B.getWidth()*A.getHeight()];

        /* Allocate device memory for the matrices */
        JCublas.cublasAlloc(A.getWidth()*A.getHeight(), Sizeof.FLOAT, d_A);
        JCublas.cublasAlloc(B.getWidth()*B.getHeight(), Sizeof.FLOAT, d_B);
        JCublas.cublasAlloc(B.getWidth()*A.getHeight(), Sizeof.FLOAT, d_C);

        /* Initialize the device matrices with the host matrices */
        JCublas.cublasSetVector(A.getWidth()*A.getHeight(), Sizeof.FLOAT, Pointer.to(h_A), 1, d_A, 1);
        JCublas.cublasSetVector(B.getWidth()*B.getHeight(), Sizeof.FLOAT, Pointer.to(h_B), 1, d_B, 1);
        JCublas.cublasSetVector(B.getWidth()*A.getHeight(), Sizeof.FLOAT, Pointer.to(h_C), 1, d_C, 1);

        int m = A.getHeight(); // A.numRows();
        int n = B.getWidth(); // B.numColumns();
        int k = A.getWidth(); // A.numColumns();

        int lda = A.getHeight(); // A.numRows();
        int ldb = B.getHeight(); // B.numRows();
        int ldc = A.getHeight(); // C.numRows();

        /* Performs operation using JCublas */
        JCublas.cublasSgemm('n', 'n', m, n, k, alpha,
                d_A, lda, d_B, ldb, beta, d_C, ldc);
        cudaDeviceSynchronize();

        /* Read the result back */
        JCublas.cublasGetVector(h_C.length, Sizeof.FLOAT, d_C, 1, Pointer.to(h_C), 1);

        /* Memory clean up */
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

//        cudaFree(d_A);
//        cudaFree(d_B);
//        cudaFree(d_C);

       // System.out.println(Arrays.toString(h_C));

        return new Matrix(arrTo2d(h_C, A.getHeight(), B.getWidth()));
    }

    public static double vectorDot(double[] hostDataA, double[] hostDataB, Pointer deviceDataA,  Pointer deviceDataB, cublasHandle handle) {

        int n = hostDataA.length;

        cudaMalloc(deviceDataA, n * Sizeof.DOUBLE);
        cudaMemcpy(deviceDataA, Pointer.to(hostDataA), n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);

        cudaMalloc(deviceDataB, n * Sizeof.DOUBLE);
        cudaMemcpy(deviceDataB, Pointer.to(hostDataB), n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);

        Pointer deviceResultPointer = new Pointer();
        cudaMalloc(deviceResultPointer, Sizeof.DOUBLE);

        cublasDdot(handle, n, deviceDataA, 1, deviceDataB, 1, deviceResultPointer);
        cudaDeviceSynchronize();

        double[] deviceResult = { -1.0 };
        cudaMemcpy(Pointer.to(deviceResult), deviceResultPointer,
                Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);

        cudaFree(deviceDataA);
        cudaFree(deviceDataB);

        return deviceResult[0];

    }

    private static void sgemmJCublas(
            int device, int n, float alpha, Pointer d_A, Pointer d_B,
            float beta, Pointer d_C)
    {
        // Execute sgemm
        Pointer pAlpha = Pointer.to(new float[]{alpha});
        Pointer pBeta = Pointer.to(new float[]{beta});
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                pAlpha, d_A, n, d_B, n, pBeta, d_C, n);
    }

//    public static Matrix matrixDot(Matrix a, Matrix b) {
//
//        cublasHandle handle = new cublasHandle();
//        cublasCreate(handle);
//
//        // Set the pointer mode to HOST
//        //cublasSetPointerMode(handle, cublasPointerMode.CUBLAS_POINTER_MODE_HOST);
//
//
//      //  int size = a.getHeight()*b.getWidth();
//
//        float[] hostInputA = getFloatArray(flatten(a));
//        float[] hostInputB = getFloatArray(flatten(b));
//        float[] hostOutput = {-1.0f};
//
//        Pointer A = Pointer.to(hostInputA);
//        Pointer B = Pointer.to(hostInputB);
//        Pointer out = Pointer.to(hostOutput);
//
////        initArray(hostInputA);
////        initArray(hostInputB);
//
//        cublasSdot(handle, a.getHeight()*b.getWidth(), A,1, B, 1, out);
//       // multiply(handle, a.getHeight()*b.getWidth(), hostInputA, hostInputB, hostOutput);
//       // return new Matrix(multiple(a, b));
//        // Matrix prod = new Matrix(arrTo2d(hostOutput, a.getHeight(), b.getWidth()));
////        System.out.println(hostOutput[0]);
//        return new Matrix(new double[][]{{hostOutput[0]}});//arrTo2d(hostOutput, hostOutput.length, 1));
//    }

    public static double[][] multiple(Matrix x, Matrix y) {
        JCublas.cublasInit();
        Pointer p_x = new Pointer();
        Pointer p_y = new Pointer();
        Pointer p_r = new Pointer();
        double [] xd = flatten(x);
        double [] yd = flatten(y);
        double [] rd = new double[1];
        JCublas.cublasAlloc(xd.length, Sizeof.DOUBLE, p_x);
        JCublas.cublasAlloc(yd.length, Sizeof.DOUBLE, p_y);
        JCublas.cublasAlloc(rd.length, Sizeof.DOUBLE, p_r);

        // Copy the memory from the host to the device
        JCublas.cublasSetVector(xd.length, Sizeof.DOUBLE, Pointer.to(xd), 1, p_x, 1);
        JCublas.cublasSetVector(yd.length, Sizeof.DOUBLE, Pointer.to(yd), 1, p_y, 1);
        JCublas.cublasSetVector(rd.length, Sizeof.DOUBLE, Pointer.to(rd), 1, p_r, 1);
        int pyRow = y.getWidth();
        int pxColumn = x.getHeight();
        int pyColumn = y.getHeight();
        int pxRow = x.getWidth();

        // Execute sgemm
        JCublas.cublasSgemm(
                'n', 'n', pyRow, pxColumn, pyColumn, 1.0f, p_y, pyRow, p_x, pxRow, 1.0f, p_r, pyRow);

        // Copy the result from the device to the host
        JCublas.cublasGetVector(rd.length, Sizeof.DOUBLE, p_r, 1, Pointer.to(rd), 1);

        // Clean up
        JCublas.cublasFree(p_x);
        JCublas.cublasFree(p_y);
        JCublas.cublasFree(p_r);

        JCublas.cublasShutdown();
        return arrTo2d(rd, x.getHeight(), y.getWidth());
    }

    public static void multiply(cublasHandle handle, int size, float A[],
                                 float B[], float C[])
    {
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, size * size * Sizeof.FLOAT);
        cudaMalloc(dB, size * size * Sizeof.FLOAT);
        cudaMalloc(dC, size * size * Sizeof.FLOAT);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, one,
                dA, size, dB, size, zero, dC, size);

        cublasGetVector(size * size, Sizeof.DOUBLE, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    static void initArray(float[] arr) {
        for(int i = 0; i<arr.length; i++) {
            arr[i] = i*((float) Math.random());
        }
    }

    public static double[][] arrTo2dColumnized(double[] arr) {
        int rowLength = (int) Math.sqrt(arr.length);
        double[][] arr2d = new double[rowLength][rowLength];
        int row = 0;
        int col = 0;
        for(int i = 0; i<arr.length; i++)
        {
            if(i % (int)Math.sqrt(arr.length) == 0 && i > 0) {
                row = 0;
                col++;
            }
            arr2d[row][col] = arr[i];
            col++;
        }
        return arr2d;
    }

    public static double[][] arrTo2d(float[] arr, int rows, int cols) {
        double[][] arr2d = new double[rows][cols];
        int row = 0;
        int col = 0;
        for(int i = 0; i<arr.length; i++)
        {
            if(i % rows == 0 && i > 0) {
                row = 0;
                col++;
            }
            arr2d[row][col] = arr[i];
            row++;
        }
        return arr2d;
    }

    public static double[][] arrTo2d(double[] arr, int rows, int cols) {
        double[][] arr2d = new double[rows][cols];
        int row = 0;
        int col = 0;
        for(int i = 0; i<arr.length; i++)
        {
            if(i % rows == 0 && i > 0) {
                row = 0;
                col++;
            }
            arr2d[row][col] = arr[i];
            row++;
        }
        return arr2d;
    }

//    public static double[][] arrTo2d(float[] arr)
//    {
//        int rowLength = (int) Math.sqrt(arr.length);
//        double[][] arr2d = new double[rowLength][rowLength];
//        int row = 0;
//        int col = 0;
//        for(int i = 0; i<arr.length; i++)
//        {
//            if(i % (int)Math.sqrt(arr.length) == 0 && i > 0) {
//                row++;
//                col = 0;
//            }
//            arr2d[row][col] = arr[i];
//            col++;
//        }
//        return arr2d;
//    }

    public static float[] getFloatArray(double[] arr) {
        float[] outArr = new float[arr.length];
        for(int i = 0; i<arr.length; i++) {
            outArr[i] = (float) arr[i];
        }
        return outArr;
    }

    private static double[] flatten(Matrix matrix) {
        return matrix.toDoubleArray();
    }
}
