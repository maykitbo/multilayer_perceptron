#include "data_manager.h"

using namespace s21;

void DataManager::ReadNoRotate(std::fstream &file, data_vector &letter)
{
    for (auto &i : letter)
    {
        if (!file)
        {
            file.close();
            throw std::runtime_error("DataManager: incorrect file");
        }
        file >> i;
        i /= 256.0;
        file.ignore();
    }
}

void DataManager::Read90Rotate(std::fstream &file, data_vector &letter)
{
    for (unsigned k = 0; k < height_; ++k)
    {
        for (unsigned g = 0; g < width_; ++g)
        {
            if (!file)
            {
                file.close();
                throw std::runtime_error("DataManager: incorrect file");
            }
            file >> letter[g * height_ + k];
            letter[g * height_ + k] /= 256.0;
            file.ignore();
        }
    }
}

auto DataManager::ReadFunctionSwitch(LetterRotate rotate)
{
    switch (rotate)
    {
        case kNoRotate:
            return &DataManager::ReadNoRotate;
        case k90Rotate:
            return &DataManager::Read90Rotate;
    }
    return &DataManager::ReadNoRotate;
}

DataManager::DataManager(const std::string &file_path, int bias, LetterRotate rotate,
                        size_t width, size_t height, unsigned classes)
    : letters_(classes)
    , width_(width)
    , height_(height)
    , classes_(classes)
{
    
    std::fstream file(file_path);

    if (!file.is_open())
        throw std::runtime_error("DataManager: file not found");

    auto read_func = ReadFunctionSwitch(rotate);
    unsigned len = width_ * height_;

    while (file)
    {
        int name = 0;
        file >> name;
        name += bias;
        if (name >= (int)classes_ || name < 0)
            break;

        file.ignore();
        letters_[name].emplace_back(len);
        (this->*read_func)(file, letters_[name].back());
        ++size_;
    }

    file.close();

    if (size_ == 0)
        throw std::runtime_error("DataManager: empty file");

    Split(Const::default_train_proportion);

}

void DataManager::ForTest(const std::function<void(data_vector&, int)> func)
{
    for (unsigned name = 0; name < classes_; ++name)
    {
        for (auto iter = letters_[name].end() - letters_[name].size() * test_proportion_;
            iter != letters_[name].end();
            ++iter)
        {
            func(*iter, name);
        }
    }
}

void DataManager::ForTrain(const std::function<void(data_vector&, int)> func)
{
    std::vector<bool> trained(classes_, false);

    for (int k = 0; k < size_ * train_proportion_; ++k)
    {
        fp_type selection = s21::Random::Uniform<fp_type>(0.0, metric_sum_);
        for (unsigned name = 0; name < classes_; ++name)
        {
            selection -= metric_[name];
            if (selection <= 0.0)
            {
                trained[name] = true;
                func(letters_[name][Random::Int<int>(0, letters_[name].size() * train_proportion_)], name);
                break;
            }
        }
    }
    for (unsigned name = 0; name < classes_; ++name)
    {
        if (!trained[name])
        {
            func(letters_[name][Random::Int<int>(0, letters_[name].size() * train_proportion_)], name);
        }
    }
}

void DataManager::Split(unsigned proportion)
{
    if (proportion == 0)
        throw std::runtime_error("DataManager: proportion == 0");

    metric_ = std::vector<fp_type>(classes_, 1.0);
    metric_sum_ = classes_;
    test_proportion_ = 1.0 / (fp_type)proportion;
    train_proportion_ = 1.0 - test_proportion_;
    cross_k_ = proportion - 1;
}

// void DataManager::SetMetric(const std::vector<fp_type> &metric) {
//     metric_sum_ = 0.0;
//     for (unsigned int k = 0; k < classes_; ++k) {
//         metric_[k] = 1.0 / metric[k];
//         metric_sum_ += metric_[k];
//     }
// }

void DataManager::Shuffle()
{
    for (auto iter = letters_.begin(); iter != letters_.end(); ++iter)
    {
        std::shuffle(iter->begin(), iter->end(),
            std::default_random_engine(std::random_device{}()));
    }
}

void DataManager::CrossUpdate()
{
    if (cross_k_ == 0)
    {
        std::cerr << "cros_k == 0\n";
        return;
    }
    for (unsigned name = 0; name < classes_; ++name)
    {
        int end = cross_k_ * letters_[name].size() * test_proportion_;
        int start = (cross_k_ - 1) * letters_[name].size() * test_proportion_;
        std::rotate(letters_[name].begin() + start, letters_[name].begin() + end, letters_[name].end());
    }
    --cross_k_;
}

void DataManager::Validate(size_t letter_size, unsigned classes)
{
    if (letter_size != width_ * height_ || classes != classes_)
        throw std::runtime_error("Validation failed: letter size or classes do not match expectations.");
}

void DataManager::PrintLetter(const data_vector &one)
{
    auto iter = one.begin();
    int s = std::sqrt(one.size());
    for (int k = s - 1; k != -1; --k)
    {
        for (int g = 0; g < s; ++g)
        {
            if (*iter > 0.2)
                printf("\033[0;31m");
            else
                printf("\033[0m");
            printf("%2.2f|", *iter);
            ++iter;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void DataManager::Printn(int n)
{
    for (int k = 0; k < n; ++k)
    {
        int l = Random::Int<int>(0, 25);
        std::cout << (char)('a' + l) << '\n';
        PrintLetter(letters_[l][Random::Int<int>(0, letters_[l].size() - 1)]);
        std::cout << '\n';
    }
}


