#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <iostream>
using namespace std;
class Tensor {
private:
    double* data_;              // memoria contigua propia
    vector<size_t> shape_; // dimensiones (máx. 3)
    size_t size_;               // producto de las dimensiones
    // Utilidades internas
    static size_t computeSize(const vector<size_t>& shape);
    void freeData();            // libera memoria y deja estado nulo
public:
    // ---- Constructor principal ----
    Tensor(const vector<size_t>& shape,const vector<double>& values);
    // ---- Regla de los cinco ----
    Tensor(const Tensor& other);                 // copia profunda
    Tensor(Tensor&& other) noexcept;             // movimiento
    Tensor& operator=(const Tensor& other);      // asignación copia
    Tensor& operator=(Tensor&& other) noexcept;  // asignación movimiento
    ~Tensor();                                   // destructor
    // ---- Creación de tensores pre-definidos ----
    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape,double min, double max);
    static Tensor arange(double start, double end);
    // ---- Operadores ----
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // elemento a elemento
    Tensor operator*(double scalar) const;       // escalar
    // ---- Modificación de dimensiones ----
    Tensor view(const vector<size_t>& newShape);
    Tensor unsqueeze(size_t dim);
    // ---- Concatenación ----
    static Tensor concat(const vector<Tensor>& tensors, size_t dim);
    // ---- Activaciones ----
    Tensor relu() const;
    Tensor sigmoid() const;
    // ---- Accesores ----
    const vector<size_t>& shape() const {
        return shape_;
    }
    size_t size() const {
        return size_;
    }
    void printShape(const string& label = "") const;
    // ---- Funciones amigas ----
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
};
#endif // TENSOR_H