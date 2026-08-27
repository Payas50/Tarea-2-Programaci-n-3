#include "Tensor.h"
#include <iostream>
#include <vector>
using namespace std;
Tensor addBias(const Tensor& x,const Tensor& bias) {
    size_t R=x.shape()[0];
    vector<Tensor> rows;
    for(size_t r=0;r<R;r++) {
        rows.push_back(bias);
    }
    Tensor expandedBias=Tensor::concat(rows,0);
    return x+expandedBias;
}
int main() {
    cout<<"Construyendo la red neuronal: "<<endl;
    Tensor input=Tensor::random({1000,20,20},0.0,1.0);
    input.printShape("Paso 1 - Entrada");
    Tensor x=input.view({1000,400});
    x.printShape("Paso 2 - Transformando utilizando View");
    Tensor W1=Tensor::random({400,100},-0.1,0.1);
    x=matmul(x,W1);
    x.printShape("Paso 3 - Multiplicando por una matriz W1");
    Tensor b1=Tensor::random({1,100},-0.1,0.1);
    x=addBias(x,b1);
    x.printShape("Paso 4 - Sumando con una matriz b1");
    x=x.relu();
    x.printShape("Paso 5 - Utilizando la funcion ReLU");
    Tensor W2=Tensor::random({100,10},-0.1,0.1);
    x=matmul(x,W2);
    x.printShape("Paso 6 - Multiplicando por una matriz W2");
    Tensor b2=Tensor::random({1,10},-0.1,0.1);
    x=addBias(x,b2);
    x.printShape("Paso 7 - Sumando con una matriz b2");
    x=x.sigmoid();
    x.printShape("Paso 8 - Utilizando la funcion Sigmoid");
    cout<<"\nSalida final lista (1000 x 10)."<<endl;
    return 0;
}