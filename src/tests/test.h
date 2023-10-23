#pragma once

#include <gtest/gtest.h>

#include "../model/graph_model.h"
#include "../model/matrix_model.h"
#include "../lib/m_sstr.h"

namespace s21 {

auto rp = std::bind(SStr::RelativePath, __FILE__, std::placeholders::_1);

PerceptronSettings CorrectSettings() {
  PerceptronSettings PS;
  PS.layers = {784, 345, 26};
  PS.learning_rate = 0.5;
  PS.SetActivation(kSigmoid);
  PS.SetWeightInit(kXavier);
  return PS;
}

} // namespace s21
