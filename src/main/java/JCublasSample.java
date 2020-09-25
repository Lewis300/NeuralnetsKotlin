import java.util.Arrays;

/* Imports, JCublas */
import jcuda.*;
import jcuda.jcublas.*;

class JCublasSample
{
    /* Main */
    public static void main(String args[])
    {
        float h_A[];
        float h_B[];
        float h_C[];
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        float alpha = 1.0f;
        float beta = 0.0f;

        /* Initialize JCublas */
        JCublas.cublasInit();

        /* Allocate host memory for the matrices */
        // 3 columns, 2 rows
        h_A = new float[] {3, 1, 2,
                1, 2, 3};
        // 2 columns, 3 rows
        h_B = new float[] {
                0, 1,
                2, 1,
                2, 3};
        // 2 columns, 2 rows
        h_C = new float[2*2];

        /* Allocate device memory for the matrices */
        JCublas.cublasAlloc(3*2, Sizeof.FLOAT, d_A);
        JCublas.cublasAlloc(2*3, Sizeof.FLOAT, d_B);
        JCublas.cublasAlloc(2*2, Sizeof.FLOAT, d_C);

        /* Initialize the device matrices with the host matrices */
        JCublas.cublasSetVector(3*2, Sizeof.FLOAT, Pointer.to(h_A), 1, d_A, 1);
        JCublas.cublasSetVector(2*3, Sizeof.FLOAT, Pointer.to(h_B), 1, d_B, 1);
        JCublas.cublasSetVector(2*2, Sizeof.FLOAT, Pointer.to(h_C), 1, d_C, 1);

        int m = 2; // A.numRows();
        int n = 2; // B.numColumns();
        int k = 3; // A.numColumns();

        int lda = 2; // A.numRows();
        int ldb = 3; // B.numRows();
        int ldc = 2; // C.numRows();

        /* Performs operation using JCublas */
        JCublas.cublasSgemm('n', 'n', m, n, k, alpha,
                d_A, lda, d_B, ldb, beta, d_C, ldc);

        /* Read the result back */
        JCublas.cublasGetVector(2*2, Sizeof.FLOAT, d_C, 1, Pointer.to(h_C), 1);

        /* Memory clean up */
        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        /* Shutdown */
        JCublas.cublasShutdown();

        System.out.println(Arrays.toString(h_C));

    }
}