#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void check(cl_int status, const std::string& operation) {
    if (status != CL_SUCCESS) {
        std::cerr << "Error en " << operation << ": OpenCL " << status << '\n';
        std::exit(EXIT_FAILURE);
    }
}

static std::string device_string(cl_device_id device, cl_device_info field) {
    size_t size = 0;
    check(clGetDeviceInfo(device, field, 0, nullptr, &size), "consultar dispositivo");
    std::string value(size, '\0');
    check(clGetDeviceInfo(device, field, size, value.data(), nullptr), "leer dispositivo");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

int main() {
    constexpr size_t N = 1'000'000;
    std::vector<float> a(N), b(N), result(N, 0.0f);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    cl_uint platform_count = 0;
    check(clGetPlatformIDs(0, nullptr, &platform_count), "buscar plataformas");
    if (platform_count == 0) {
        std::cerr << "No se encontraron plataformas OpenCL. Verifica el runtime del fabricante.\n";
        return EXIT_FAILURE;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "obtener plataformas");

    cl_device_id device = nullptr;
    for (auto platform : platforms) {
        cl_uint count = 0;
        cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
        if (status == CL_SUCCESS && count > 0) {
            std::vector<cl_device_id> devices(count);
            check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr),
                  "obtener GPU");
            device = devices.front();
            break;
        }
    }

    if (!device) {
        std::cerr << "No se detectó una GPU OpenCL. No se usará CPU como sustituto.\n";
        std::cerr << "Comprueba el driver/runtime OpenCL del fabricante y el acceso a /dev/dri.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Dispositivo OpenCL: " << device_string(device, CL_DEVICE_NAME) << '\n';
    std::cout << "Fabricante: " << device_string(device, CL_DEVICE_VENDOR) << '\n';
    std::cout << "Problema: suma de vectores; elementos: " << N << "\n";

    const char* source = R"CLC(
        __kernel void vector_add(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const unsigned int n
        ) {
            const size_t i = get_global_id(0);
            if (i < n) c[i] = a[i] + b[i];
        }
    )CLC";

    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    check(status, "crear contexto");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    check(status, "crear cola");

    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &status);
    check(status, "crear programa");
    status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Error al compilar el kernel:\n" << log << '\n';
        return EXIT_FAILURE;
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &status);
    check(status, "crear kernel");

    cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * N, a.data(), &status);
    check(status, "crear buffer A");
    cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * N, b.data(), &status);
    check(status, "crear buffer B");
    cl_mem buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      sizeof(float) * N, nullptr, &status);
    check(status, "crear buffer C");

    const cl_uint n = static_cast<cl_uint>(N);
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a), "argumento A");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b), "argumento B");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c), "argumento C");
    check(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n), "argumento N");

    const size_t global_size = N;
    cl_event event;
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr,
                                 0, nullptr, &event), "ejecutar kernel");
    check(clEnqueueReadBuffer(queue, buffer_c, CL_TRUE, 0, sizeof(float) * N,
                              result.data(), 1, &event, nullptr), "leer resultado");

    cl_ulong start = 0, end = 0;
    check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
          "tiempo inicial");
    check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
          "tiempo final");

    size_t errors = 0;
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(result[i] - (a[i] + b[i])) > 1e-5f) ++errors;
    }

    std::cout << "Ejemplo: C[i] = A[i] + B[i]\n";
    std::cout << "C[0] = " << result[0] << ", C[1] = " << result[1]
              << ", C[N-1] = " << result[N - 1] << '\n';
    std::cout << "Tiempo del kernel: " << (end - start) / 1'000'000.0 << " ms\n";
    std::cout << "Validación: " << (errors == 0 ? "CORRECTA" : "FALLIDA")
              << " (" << errors << " errores)\n";

    clReleaseEvent(event);
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
