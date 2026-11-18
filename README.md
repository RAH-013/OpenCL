# OpenCL + C++: suma paralela de vectores

Ejemplo mínimo para demostrar cómo un kernel OpenCL ejecuta una operación matemática en una GPU.

## Problema

Dados dos vectores `A` y `B`, calcular:

`C[i] = A[i] + B[i]`

El ejemplo procesa 1,000,000 de elementos. Cada work-item ejecuta la suma de una posición independiente.

## Requisitos del host

- Docker Engine o Podman con Compose.
- Driver gráfico instalado.
- Runtime OpenCL del fabricante compatible con la GPU.
- En Linux, acceso a `/dev/dri` para GPUs AMD/Intel.

El contenedor incluye el cargador OpenCL, encabezados, herramientas y Mesa OpenCL. El runtime del fabricante y el acceso al dispositivo siguen dependiendo del host. Para NVIDIA normalmente se necesita NVIDIA Container Toolkit y exponer la GPU al contenedor.

## Ejecutar con Docker Compose

```bash
docker compose up --build
```

Para Podman:

```bash
podman compose up --build
```

También se puede inspeccionar el runtime dentro del contenedor:

```bash
docker compose run --rm opencl-demo clinfo
```

## Ejecutar con Parrot OS / Debian y AMD

Comprueba primero en el host:

```bash
ls -l /dev/dri
clinfo
```

Si `clinfo` no existe, instala las herramientas y el runtime OpenCL de tu distribución. El compose configura `RUSTICL_ENABLE=radeonsi`, usado por Mesa Rusticl para GPUs AMD compatibles. Si tu sistema utiliza otro runtime, ajusta o elimina esa variable.

Si aparece `No se detectó una GPU OpenCL`, el programa no cambia silenciosamente a CPU: indica que falta configurar el runtime o el acceso al dispositivo.

## Qué explicar en la exposición

1. **Host:** prepara los datos y administra la ejecución.
2. **Kernel:** función que se ejecuta en paralelo en los work-items de la GPU.
3. **Global ID:** `get_global_id(0)` identifica qué índice procesa cada work-item.
4. **Memoria:** los buffers trasladan los vectores entre el host y el dispositivo.
5. **Lanzamiento:** `clEnqueueNDRangeKernel` distribuye el trabajo.
6. **Validación:** se compara el resultado de cada posición con la suma esperada.
7. **Medición:** el evento OpenCL mide el tiempo de ejecución del kernel, no el tiempo total de transferencias ni la inicialización.

## Comparación justa entre equipos

- Registrar modelo de GPU, fabricante, driver y runtime OpenCL.
- Mantener el mismo tamaño de vectores y tipo de dato.
- Separar tiempo de kernel del tiempo de transferencias.
- Repetir varias ejecuciones y reportar promedio/mediana.
- No concluir que una arquitectura es mejor con una sola ejecución: influyen drivers, memoria, clocks y configuración.
