#include "test.h"

using namespace s21;


TEST(Settings, settings_constructor_1) {
  EXPECT_NO_THROW(MatrixModel MM(CorrectSettings()));
  EXPECT_NO_THROW(GraphModel GM(CorrectSettings()));
}

TEST(Settings, settings_constructor_2) {
  auto ps = CorrectSettings();
  ps.SetWeightInit(kNormal);
  EXPECT_NO_THROW(MatrixModel MM(ps));
  EXPECT_NO_THROW(GraphModel GM(ps));
}

void IncorrectSettingsTest(std::function<void(PerceptronSettings&)> set) {
  auto PS = CorrectSettings();
  set(PS);
  EXPECT_ANY_THROW(MatrixModel MM(PS));
}

TEST(Settings, error_settings_constructor_1) {
  IncorrectSettingsTest(
    [] (PerceptronSettings &PS) { PS.layers = {}; });
  IncorrectSettingsTest(
    [] (PerceptronSettings &PS) { PS.learning_rate = 0.0; });
  IncorrectSettingsTest(
    [] (PerceptronSettings &PS) { PS.learning_rate = -1.9; });
  IncorrectSettingsTest(
    [] (PerceptronSettings &PS) { PS.lr_epoch_k = -0.1; });
  IncorrectSettingsTest(
    [] (PerceptronSettings &PS) { PS.lr_layers_k = -0.4; });
}

TEST(Settings, error_settings_constructor_2) {
  PerceptronSettings PS;
  PS.layers = {784, 345, 26};
  PS.learning_rate = 0.5;
  PS.SetActivation(kSigmoid);
  EXPECT_ANY_THROW(MatrixModel MM(PS));
}

TEST(Settings, error_settings_constructor_3) {
  PerceptronSettings PS;
  PS.layers = {784, 345, 26};
  PS.learning_rate = 0.5;
  PS.SetWeightInit(kXavier);
  EXPECT_ANY_THROW(MatrixModel MM(PS));
}
