#include <iostream>
#include "NN.hpp"
#include <math.h>


float activation(float x) {
    return tanhf(x);
}

float activation_derivate(float x) {
    return 1 - tanhf(x) * tanhf(x); 
}

float fx(float x) {
    return std::pow(x, 2);
}

int main(int argc, char* argv[]) 
{

    int samples = atoi(argv[1]);
    double lr = atof(argv[2]);
    int epochs = atof(argv[3]);
    int pixels = atoi(argv[4]);
    float a = -1;
    float b = 1;
    float h = (b-a)/(samples-1);
    int i;
    int j = 0;
    float h_i;    
    
    ColVector* x;
    ColVector* pred;
    Index max_i;
    std::vector<ColVector*> X;
    std::vector<ColVector*> y;
    ColVector* red;
    ColVector* blue;
    ColVector* y_red;
    ColVector* y_blue;

    std::cout << "Samples: " << samples;
    std::cout << ", Learning Rate: " << lr;
    std::cout << ", Epochs: " << epochs;
    std::cout << ", Pixels output: " << pixels << std::endl << std::endl;

    // Creating samples
    for (i = 0; i < samples; i++) {
        red = new ColVector(2);
        y_red = new ColVector(2);
        (*red)[0] = a + i*h;
        (*red)[1] = fx((*red)[0]) - 0.6;
        (*y_red)[0] = 1;
        (*y_red)[1] = 0;

        blue = new ColVector(2);
        y_blue = new ColVector(2);
        (*blue)[0] = a + i*h;
        (*blue)[1] = fx((*blue)[0]);
        (*y_blue)[0] = 0;
        (*y_blue)[1] = 1;


        X.push_back(blue);
        y.push_back(y_blue);

        X.push_back(red);
        y.push_back(y_red);
    }



    // training
    NN n({2, 2, 2});
    n.train(X, y, lr, epochs);

    std::cout << std::endl;

    // Exibir matriz das layers;
    for (i = 0; i < n.weights.size(); i++) {
        std::cout << " Layer: " << i+1 << std::endl;
        std::cout << *n.weights[i] << std::endl;
        std::cout << " Bias: " << i+1 << std::endl;
        std::cout << *n.bias[i] << std::endl << std::endl;
    }

    std::cout << std::endl << " Imagem: " << std::endl;


    x = new ColVector(2);
    h_i = 1.6/ (pixels - 1);
    h = (b - a) / (pixels - 1);

    for (i = 0; i < pixels; i++) {
        // std::cout << 2 - i*h_i << " ";
        for (j = 0; j < pixels; j++) {
            (*x)[0] = a + j*h;
            (*x)[1] = 1 - i*h_i;

            pred = n.predict(x);
            pred->maxCoeff(&max_i);

            std::cout << max_i << " ";
        }
        std::cout << std::endl;
    }


    // Limpar memoria
    delete x;

    for (i = 0; i < X.size(); i++) {
        delete X[i];
        delete y[i];
    }
}