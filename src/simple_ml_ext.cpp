#include <algorithm>
#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int start = 0; start < m; start += batch) {
      int batch_size = std::min(batch, m - start);
      float *Z = new float[batch_size * k]; // Z : shape (m, k)
      const float *X_batch = X + start * n;
      const unsigned char *y_batch = y + start;

      // step1, calculate for Z

      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < k; j++) {
          Z[i * k + j] = 0;
          for (int a = 0; a < n; a++) {
            Z[i * k + j] += X_batch[i * n + a] * theta[a * k + j];
          }
        }
      }

      // step2 calculate max number in each row of Z and substract from it

      for (int i = 0; i < batch_size; i++) {
        float max_number = Z[i * k];
        for (int j = 0; j < k; j++) {
          max_number = std::max(max_number, Z[i * k + j]);
        }
        for (int j = 0; j < k; j++) {
          Z[i * k + j] -= max_number;
        }
      }

      // step 3 calculate softmax for Z

      for (int i = 0; i < batch_size; i++) {
        float sum = 0;
        for (int j = 0; j < k; j++) {
          sum += std::exp(Z[i * k + j]);
        }
        for (int j = 0; j < k; j++) {
          Z[i * k + j] = std::exp(Z[i * k + j]) / sum;
        }
      }

      // step4 calculate the one hot for y

      int *Iy = new int[batch * k];

      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < k; j++) {
          if (j == y_batch[i])
            Iy[i * k + j] = 1;
          else
            Iy[i * k + j] = 0;
        }
      }

      // step5 caculate the grad

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
          float grad = 0;
          for (int f = 0; f < batch_size; f++) {
            grad += X_batch[f * n + i] * (Z[f * k + j] - Iy[f * k + j]);
          }
          theta[i * k + j] -= lr * grad / batch_size;
        }
      }

      delete[] Z;
      delete[] Iy;
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
