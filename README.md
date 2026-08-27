# Tarea-2-Programacion-3
# Archivos:
- `Tensor.h`: Declaración de la clase Tensor.
- `Tensor.cpp`: Implementación de la clase Tensor.
- `main.cpp`: Programa principal y pruebas.
- `README.md`: Instrucciones de compilación y ejecución.
# Ejecución:
- Abrir el proyecto en CLion.
- Verificar que Tensor.h, Tensor.cpp y main.cpp estén dentro del proyecto y del `CMakeList`.
- Ejecutar el programa desde el botón Run.
# Funcionamiento:
El programa realiza pruebas de las operaciones implementadas en la clase Tensor:  
Creación de tensores con zeros.  
Creación de tensores con ones.  
Generación de valores aleatorios con random.  
Generación de secuencias con arange.  
Cambio de dimensiones con view.  
Agregar una dimensión con unsqueeze.  
Suma de tensores.  
Resta de tensores.  
Multiplicación elemento a elemento.  
Multiplicación por escalar.  
Concatenación con concat.  
Producto punto con dot.  
Multiplicación de matrices con matmul.  
Función ReLU.  
Función Sigmoid.  
También se realiza una prueba de una red neuronal utilizando tensores de dimensiones:

Entrada:        (1000, 20, 20)  
View:            (1000, 400)  
Matmul W1:      (1000, 100)  
Bias 1:          (1000, 100)  
ReLU:            (1000, 100)  
Matmul W2:      (1000, 10)  
Bias 2:          (1000, 10)  
Sigmoid:         (1000, 10)  
# Resultado esperado:
Al ejecutar el programa, se muestran por consola las dimensiones (shape) de los tensores después de cada operación.  
El programa finaliza mostrando:  
Salida final lista (1000 x 10).
