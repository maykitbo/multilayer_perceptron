#include "model.h"

using namespace s21;

void Model::Train(DataManager &inputs, unsigned epoch_count, unsigned batch,
        bool &stop, bool error_thread, bool metric_thread)
{

    inputs.Validate(settings_.layers.front(), settings_.layers.back());
    if (error_thread)
        error_.SetDatasetSize(inputs.TrainSize());

    auto time_point = std::chrono::high_resolution_clock::now();
    unsigned batch_count = 0;
    for (unsigned int k = 0; k < epoch_count; ++k)
    {
        if (error_thread)
            error_.SetEpoch(k + 1);

        inputs.ForTrain([&] (data_t &input)
        {
            if (stop)
                return;
            
            Forward(input.first);
            Backward(input);
            if (error_thread)
                error_.Collect(GetMeanError());

            if (++batch_count == batch)
            {
                UpdateWeights();
                batch_count = 0;
            }
        }, batch);
        if (stop)
            return;

        if (metric_thread)
            EpochTest(inputs, Time::Duration<Time::ms>(time_point));

        UpdateLR();
    }
    error_.Clear();
    
}

Metrics Model::Test(DataManager &inputs)
{
    auto time_point = std::chrono::high_resolution_clock::now();

    inputs.Validate(settings_.layers.front(), settings_.layers.back());

    MetricsMaker metrics(settings_.layers.back());

    inputs.ForTest([&] (data_t &input)
    {
        // letter_ = &letter.first;
        Forward(input.first);
        metrics(GetResult(), input.second);
    });
    auto metr = metrics();
    metr.testing_time = Time::Duration<Time::ms>(time_point);
    return metr;
}

void Model::EpochTest(DataManager &inputs, int64_t training_time)
{
    auto epoch_metrics = Test(inputs);

    epoch_metrics.training_time = training_time;
    epoch_metrics.testing_time = epoch_metrics.testing_time;

    metrics_func_(epoch_metrics);
}

char Model::Predict(data_vector &input)
{
    if (input.size() != settings_.layers.front())
        throw std::runtime_error("Model: incorrect letter size");

    // letter_ = &letter;
    Forward(input);
    return GetResult() + 'A';
}

void Model::SetErrorThread(const std::function<void(fp_type, unsigned)> &func)
{
    error_.SetFunc(func);
}

void Model::SetMetricThread(const std::function<void(Metrics)> &func)
{
    metrics_func_ = func;
}

