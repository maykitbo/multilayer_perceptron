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
        i = (i / (255.0 / 2.0)) - 1.0;
        // i = i / 255.0;
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
            fp_type temp;
            file >> temp;
            temp = (temp / (255.0 / 2.0)) - 1.0;
            // temp = temp / 255.0;
            letter[g * height_ + k] = temp;
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
    : width_(width)
    , height_(height)
    , classes_(classes)
    , test_proportion_(0.0)
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
        letters_.emplace_back(std::vector<fp_type>(len), name);
        (this->*read_func)(file, letters_.back().first);
        ++size_;
    }

    file.close();

    if (size_ == 0)
        throw std::runtime_error("DataManager: empty file");

    Shuffle();
    Split(Const::default_train_proportion);

}

void DataManager::ForTest(const std::function<void(data_t&)> func)
{
    for (auto iter = letters_.end() - size_ * test_proportion_; iter != letters_.end(); ++iter)
    {
        func(*iter);
    }
    
}

void DataManager::ForTrain(const std::function<void(data_t&)> func, int batch)
{
    Shuffle();
    int n = ((int)(size_ * train_proportion_) / batch) * batch;
    for (int k = 0; k < n; ++k)
    {
        func(letters_[k]);
    }
}

void DataManager::Split(unsigned proportion)
{
    if (proportion == 0)
        throw std::runtime_error("DataManager: proportion == 0");

    test_proportion_ = 1.0 / (fp_type)proportion;
    train_proportion_ = 1.0 - test_proportion_;
    cross_k_ = proportion - 1;
}

void DataManager::Shuffle()
{
    std::shuffle(letters_.begin(), letters_.end() - size_ * test_proportion_,
        std::default_random_engine(std::random_device{}()));

}

void DataManager::CrossUpdate()
{
    if (cross_k_ == 0)
    {
        std::cerr << "cros_k == 0\n";
        return;
    }
    std::rotate(letters_.begin(), letters_.end() - size_ * test_proportion_, letters_.end());
    --cross_k_;
}

void DataManager::Validate(size_t letter_size, unsigned classes)
{
    if (letter_size != width_ * height_ || classes != classes_)
        throw std::runtime_error("Validation failed: letter size or classes do not match expectations.");
}

void DataManager::IncreaseRoatation(int angle)
{
    letters_.resize(size_ * 3);
    for (unsigned k = 0; k < size_; ++k)
    {
        letters_[size_ * 2 + k] = letters_[k];
        letters_[size_ + k].first = Rotate(letters_[k].first, -angle);
        letters_[k].first = Rotate(letters_[k].first, angle);
    }
    size_ *= 3;
    Shuffle();
}

std::vector<fp_type> DataManager::Rotate(const std::vector<fp_type> &source, int angle)
{
    double radians = angle * M_PI / 180.0;
    double cos_t = cos(radians);
    double sin_t = sin(radians);

    int size = sqrt(source.size());
    int cx = size / 2;
    int cy = size / 2;

    std::vector<double> rotated(size * size, 0.0);

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int rx = static_cast<int>((x - cx) * cos_t - (y - cy) * sin_t + cx);
            int ry = static_cast<int>((x - cx) * sin_t + (y - cy) * cos_t + cy);

            if (rx >= 0 && rx < size && ry >= 0 && ry < size) {
                rotated[y * size + x] = source[ry * size + rx];
            }
        }
    }

    return rotated;
}



void DataManager::PrintLetter(const data_t &one)
{
    std::cout << one.second << '\n';
    auto iter = one.first.begin();
    int s = std::sqrt(one.first.size());
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
    // for (int k = 0; k < n; ++k)
    // {
    //     int l = Random::Int<int>(0, 25);
    //     std::cout << (char)('a' + l) << '\n';
    //     PrintLetter(letters_[l][Random::Int<int>(0, letters_[l].size() - 1)]);
    //     std::cout << '\n';
    // }
}


