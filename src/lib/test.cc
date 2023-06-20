
#include <map>

#include <algorithm>
#include <iostream>
#include <vector>

#include "m_time.h"
#include "random.h"

int main() {
    const int size = 1e10;

    auto time_point = s21::Time::Now();
    std::vector<double> V(1e10);
    for (int k = 0; k < size; ++k) {
        V[k] = std::sqrt(size);
    }
    std::cout << "create vector :" << s21::Time::Duration(time_point) << '\n';

    time_point = s21::Time::Now();
    std::map<int, double> M(size);
    

    std::cout << "create vector :" << s21::Time::Duration(time_point) << '\n';


    auto result = s21::Time::Compare(100000, [
        int k = s21::Random::Int(0, size - 1);
        // V[k] = std::pow(V[k], 2.4);
        V[k] = std::pow(k, 2.4);
    ], [] {
        int k = s21::Random::Int(0, size - 1);
        // V[k] = std::pow(M[k], 2.4);
        M[k] = std::pow(k, 2.4);
    });

    std::cout << result[1] << " " << result[2] << '\n';

    return 0;
}
