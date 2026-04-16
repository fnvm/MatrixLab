package com.github.fnvm.MatrixLab;

import org.jblas.DoubleMatrix;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/*
 * Для пропуска наивного алгоритма запустить с аргументом -s
 * Иначе он будет считать вечно...
 */
public class MatrixBenchmark {

  private static final int N = 4096;
  private static final double complexity = 2.0 * Math.pow(N, 3);
  // Для собственной реализации:
  private static final int TILE = 64;
  private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();

  public static void main(String[] args) {
    boolean skip = false;
    if (args.length > 0 && args[0].trim().equals("-s")) {
      skip = true;
    }

    System.out.printf("Умножение матриц  N=%d %n", N);
    System.out.printf("Доступные процессоры: %d%n%n", PARALLELISM);

    System.out.printf("Теоретическая сложность c = 2·N³ = %.3e %n%n", complexity);

    // Создание матриц
    double[] A = generateRandom(N);
    double[] B = generateRandom(N);

    double[] res = new double[2];
    if (!skip) {
      // Вариант 1
      System.out.println("Вариант 1: наивный алгоритм");

      long t1Start = System.nanoTime();
      double[] C1 = multiplyNaive(A, B, N);
      long t1 = System.nanoTime() - t1Start;

      res = getResult(t1, true);
    }

    // Вариант 2
    System.out.println("Вариант 2: BLAS");
    // JBLAS хранит данные по столбцам (Fortran-order), поэтому транспонируем
    DoubleMatrix mA = new DoubleMatrix(N, N, A).transpose();
    DoubleMatrix mB = new DoubleMatrix(N, N, B).transpose();

    long t2Start = System.nanoTime();
    DoubleMatrix mC = mA.mmul(mB);
    long t2 = System.nanoTime() - t2Start;

    double res2[] = getResult(t2, true);
    double sec2 = res2[0];
    double mflops2 = res2[1];

    // Вариант 3
    System.out.println("Вариант 3: блочная реализация");

    long t3Start = System.nanoTime();
    double[] C3 = multiplyTiledParallel(A, B, N);
    long t3 = System.nanoTime() - t3Start;

    double[] res3 = getResult(t3, true);
    double sec3 = res3[0];
    double mflops3 = res3[1];

    if (!skip) {
      double sec1 = res[0];
      double mflops1 = res[1];

      System.out.printf(
          "1. Наивный алгоритм. Время: %13.3f с. | MFlops: %14.1f | от BLAS: %10.1f %%%n",
          sec1, mflops1, mflops1 / mflops2 * 100);
    }
    System.out.printf("2. BLAS. Время: %13.3f с. | MFlops: %14.1f %n", sec2, mflops2);
    System.out.printf(
        "3. Блочный. Время: %13.3f с. | MFlops: %14.1f | от BLAS: %10.1f %%%n",
        sec3, mflops3, mflops3 / mflops2 * 100);
  }

  // Наивный алгоритм перемножения матриц
  public static double[] multiplyNaive(double[] A, double[] B, int n) {
    double[] C = new double[n * n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
          sum += A[i * n + k] * B[k * n + j];
        }
        C[i * n + j] = sum;
      }
    }
    return C;
  }

  // Вариант 3
  public static double[] multiplyTiledParallel(double[] A, double[] B, int n) {
    double[] Bt = transpose(B, n);
    double[] C = new double[n * n];

    // Делим строки матрицы A между потоками
    ForkJoinPool pool = new ForkJoinPool(PARALLELISM);

    pool.submit(
            () ->
                IntStream.range(0, (n + TILE - 1) / TILE)
                    .parallel()
                    .forEach(
                        ti -> {
                          int iStart = ti * TILE;
                          int iEnd = Math.min(iStart + TILE, n);

                          for (int tj = 0; tj < n; tj += TILE) {
                            int jEnd = Math.min(tj + TILE, n);

                            for (int tk = 0; tk < n; tk += TILE) {
                              int kEnd = Math.min(tk + TILE, n);

                              for (int i = iStart; i < iEnd; i++) {
                                int rowA = i * n + tk;
                                for (int j = tj; j < jEnd; j++) {
                                  int rowBt = j * n + tk;
                                  double sum = C[i * n + j];
                                  for (int k = 0; k < kEnd - tk; k++) {
                                    sum += A[rowA + k] * Bt[rowBt + k];
                                  }
                                  C[i * n + j] = sum;
                                }
                              }
                            }
                          }
                        }))
        .join();

    pool.shutdown();
    pool.close();
    return C;
  }

  // Генерация случайной матрицы n×n
  public static double[] generateRandom(int n) {
    Random rnd = new Random(42L);
    double[] M = new double[n * n];
    for (int i = 0; i < M.length; i++) {
      M[i] = rnd.nextDouble() * 2.0 - 1.0; // [-1, 1)
    }
    return M;
  }

  // Транспонирование матрицы n×n
  public static double[] transpose(double[] M, int n) {
    double[] T = new double[n * n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        T[j * n + i] = M[i * n + j];
      }
    }
    return T;
  }

  private static double[] getResult(long nanoseconds, boolean print) {
    double sec = nanoseconds * 1e-9;
    double mflops = complexity / sec * 1e-6;

    if (print) {
      System.out.printf("Время:          %.3f с%n", sec);
      System.out.printf("Производительность: %.1f MFlops%n%n", mflops);
    }

    return new double[] {sec, mflops};
  }
}
