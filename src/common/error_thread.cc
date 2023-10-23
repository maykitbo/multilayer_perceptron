#include "error_thread.h"

using namespace s21;

void Error::SetFunc(const std::function<void(fp_type, unsigned int)> &f) {
    func_ = f;
}

void Error::SetEpoch(unsigned int epoch) {
    epoch_ = epoch;
    Clear();
}

void Error::SetDatasetSize(unsigned int size) {
    period_ = size / (Const::error_updates);
}

void Error::Clear() {
    value_ = 0.0;
    count_ = 0;
}

void Error::Collect(fp_type error) {
    value_ *= count_;
    value_ += error;
    value_ /= ++count_;

    if (count_ == period_)
        Send();
}

void Error::Send() {
    func_(value_, epoch_);
    Clear();
}
