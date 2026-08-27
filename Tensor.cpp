#include "Tensor.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
using namespace std;
size_t Tensor::computeSize(const vector<size_t>& shape) {
    if(shape.empty()) return 0;
    size_t s=1;
    for(size_t i=0;i<shape.size();i++) {
        s=s*shape[i];
    }
    return s;
}
void Tensor::freeData() {
    delete[] data_;
    data_=nullptr;
    size_=0;
    shape_.clear();
}
Tensor::Tensor(const vector<size_t>& shape,const vector<double>& values) {
    shape_=shape;
    size_=computeSize(shape);
    data_=new double[size_];
    for(size_t i=0;i<size_;i++) {
        data_[i]=values[i];
    }
}
Tensor::Tensor(const Tensor& other) {
    shape_=other.shape_;
    size_=other.size_;
    data_=new double[size_];
    for(size_t i=0;i<size_;i++) {
        data_[i]=other.data_[i];
    }
}
Tensor::Tensor(Tensor&& other) noexcept {
    data_=other.data_;
    shape_=other.shape_;
    size_=other.size_;
    other.data_=nullptr;
    other.size_=0;
    other.shape_.clear();
}
Tensor& Tensor::operator=(const Tensor& other) {
    if(this!=&other) {
        freeData();
        shape_=other.shape_;
        size_=other.size_;
        data_=new double[size_];
        for(size_t i=0;i<size_;i++) {
            data_[i]=other.data_[i];
        }
    }
    return *this;
}
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if(this!=&other) {
        freeData();
        data_=other.data_;
        shape_=other.shape_;
        size_=other.size_;
        other.data_=nullptr;
        other.size_=0;
        other.shape_.clear();
    }
    return *this;
}
Tensor::~Tensor() {
    freeData();
}
Tensor Tensor::zeros(const vector<size_t>& shape) {
    size_t n=computeSize(shape);
    vector<double> values;
    for(size_t i=0;i<n;i++) {
        values.push_back(0);
    }
    return Tensor(shape,values);
}
Tensor Tensor::ones(const vector<size_t>& shape) {
    size_t n=computeSize(shape);
    vector<double> values;
    for(size_t i=0;i<n;i++) {
        values.push_back(1);
    }
    return Tensor(shape,values);
}
Tensor Tensor::random(const vector<size_t>& shape,double min,double max) {
    size_t n=computeSize(shape);
    vector<double> values;
    for(size_t i=0;i<n;i++) {
        double numero=rand()/(RAND_MAX+1.0);
        values.push_back(min+numero*(max-min));
    }
    return Tensor(shape,values);
}
Tensor Tensor::arange(double start,double end) {
    vector<double> values;
    for(double i=start;i<end;i++) {
        values.push_back(i);
    }
    vector<size_t> shape;
    shape.push_back(values.size());
    return Tensor(shape,values);
}
Tensor Tensor::operator+(const Tensor& other) const {
    if(shape_!=other.shape_) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        values.push_back(data_[i]+other.data_[i]);
    }
    return Tensor(shape_,values);
}
Tensor Tensor::operator-(const Tensor& other) const {
    if(shape_!=other.shape_) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        values.push_back(data_[i]-other.data_[i]);
    }
    return Tensor(shape_,values);
}
Tensor Tensor::operator*(const Tensor& other) const {
    if(shape_!=other.shape_) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        values.push_back(data_[i]*other.data_[i]);
    }
    return Tensor(shape_,values);
}
Tensor Tensor::operator*(double scalar) const {
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        values.push_back(data_[i]*scalar);
    }
    return Tensor(shape_,values);
}
Tensor Tensor::view(const vector<size_t>& newShape) {
    if(computeSize(newShape)!=size_) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        values.push_back(data_[i]);
    }
    return Tensor(newShape,values);
}
Tensor Tensor::unsqueeze(size_t dim) {
    if(dim>shape_.size()) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    vector<size_t> newShape=shape_;
    newShape.insert(newShape.begin()+dim,1);
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        values.push_back(data_[i]);
    }
    return Tensor(newShape,values);
}
Tensor Tensor::concat(const vector<Tensor>& tensors,size_t dim) {
    if(tensors.empty()) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    if(dim>=tensors[0].shape_.size()) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    vector<size_t> newShape=tensors[0].shape_;
    size_t total=tensors[0].shape_[dim];
    for(size_t i=1;i<tensors.size();i++) {
        if(tensors[i].shape_.size()!=newShape.size()) {
            return Tensor(vector<size_t>{0},vector<double>{});
        }
        for(size_t j=0;j<newShape.size();j++) {
            if(j!=dim && tensors[i].shape_[j]!=newShape[j]) {
                return Tensor(vector<size_t>{0},vector<double>{});
            }
        }
        total=total+tensors[i].shape_[dim];
    }
    newShape[dim]=total;
    vector<double> values;
    for(size_t i=0;i<tensors.size();i++) {
        for(size_t j=0;j<tensors[i].size_;j++) {
            values.push_back(tensors[i].data_[j]);
        }
    }
    return Tensor(newShape,values);
}
Tensor Tensor::relu() const {
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        if(data_[i]>0) {
            values.push_back(data_[i]);
        } else {
            values.push_back(0);
        }
    }
    return Tensor(shape_,values);
}
Tensor Tensor::sigmoid() const {
    vector<double> values;
    for(size_t i=0;i<size_;i++) {
        values.push_back(1.0/(1.0+exp(-data_[i])));
    }
    return Tensor(shape_,values);
}
void Tensor::printShape(const string& label) const {
    if(label!="") {
        cout<<label<<" ";
    }
    cout<<"Forma: (";
    for(size_t i=0;i<shape_.size();i++) {
        cout<<shape_[i];
        if(i+1<shape_.size()) {
            cout<<", ";
        }
    }
    cout<<") Tamanio: "<<size_<<endl;
}
Tensor dot(const Tensor& a,const Tensor& b) {
    if(a.shape_.size()!=1 || b.shape_.size()!=1) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    if(a.size_!=b.size_) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    double resultado=0;
    for(size_t i=0;i<a.size_;i++) {
        resultado=resultado+a.data_[i]*b.data_[i];
    }
    return Tensor(vector<size_t>{1},vector<double>{resultado});
}
Tensor matmul(const Tensor& a,const Tensor& b) {
    if(a.shape_.size()!=2 || b.shape_.size()!=2) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    if(a.shape_[1]!=b.shape_[0]) {
        return Tensor(vector<size_t>{0},vector<double>{});
    }
    size_t filas=a.shape_[0];
    size_t comun=a.shape_[1];
    size_t columnas=b.shape_[1];
    vector<double> values;
    for(size_t i=0;i<filas;i++) {
        for(size_t j=0;j<columnas;j++) {
            double resultado=0;
            for(size_t k=0;k<comun;k++) {
                resultado=resultado+
                    a.data_[i*comun+k]*
                    b.data_[k*columnas+j];
            }
            values.push_back(resultado);
        }
    }
    vector<size_t> shape;
    shape.push_back(filas);
    shape.push_back(columnas);
    return Tensor(shape,values);
}