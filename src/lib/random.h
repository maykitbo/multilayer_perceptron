#pragma once

#include <random>
#include <map>
#include <iostream>

namespace s21 {

namespace Random {

template<class Type, class Distribution, class Engine, class ...Args>
Type Full(Args &&...args) {
    static Engine generator{std::random_device{}()};
    static std::map<std::tuple<std::decay_t<Args>...>, Distribution> distributions;
    return distributions.emplace(std::tie(args...), Distribution(args...)).first->second(generator);
}

template<class FPType = float, class Distribution = std::normal_distribution<FPType>, class Engine = std::mt19937>
FPType Normal(FPType mean, FPType sd) {
    return Full<FPType, Distribution, Engine>(mean, sd);
}

template<class Distribution = std::bernoulli_distribution, class Engine = std::mt19937>
bool Bool(double chance) {
    return Full<bool, Distribution, Engine>(chance);
}

template<class FPType = float, class Engine = std::mt19937>
FPType Uniform(FPType min, FPType max = 1.0) {
    return Full<FPType, std::uniform_real_distribution<FPType>, Engine>(min, max);
}

template<class IntType = int, class Engine = std::mt19937>
IntType Int(IntType min, IntType max = std::numeric_limits<IntType>::max()) {
    return Full<IntType, std::uniform_int_distribution<IntType>, Engine>(min, max);
}

} // namespace Random

} // namespace s21