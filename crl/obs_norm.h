#pragma once
#include <vector>
#include <cmath>
#include <cstring>

class RunningObsNorm {
public:
    int size_;
    float clip_;
    float eps_;
    std::vector<double> mean_;
    std::vector<double> var_;
    double count_;

    RunningObsNorm() : size_(0), clip_(10.0f), eps_(1e-8f), count_(1e-4) {}

    void init(int size, float clip = 10.0f, float eps = 1e-8f) {
        size_ = size;
        clip_ = clip;
        eps_ = eps;
        mean_.assign(size, 0.0);
        var_.assign(size, 1.0);
        count_ = 1e-4;
    }

    void update(const float* batch, int batch_size) {
        if (batch_size <= 0 || size_ <= 0) return;
        std::vector<double> batch_mean(size_, 0.0);
        std::vector<double> batch_var(size_, 0.0);

        for (int i = 0; i < batch_size; i++) {
            const float* row = batch + i * size_;
            for (int j = 0; j < size_; j++) {
                batch_mean[j] += (double)row[j];
            }
        }
        for (int j = 0; j < size_; j++) {
            batch_mean[j] /= batch_size;
        }

        for (int i = 0; i < batch_size; i++) {
            const float* row = batch + i * size_;
            for (int j = 0; j < size_; j++) {
                double d = (double)row[j] - batch_mean[j];
                batch_var[j] += d * d;
            }
        }
        for (int j = 0; j < size_; j++) {
            batch_var[j] /= batch_size;
        }

        double total = count_ + (double)batch_size;
        for (int j = 0; j < size_; j++) {
            double delta = batch_mean[j] - mean_[j];
            double new_mean = mean_[j] + delta * batch_size / total;
            double m_a = var_[j] * count_;
            double m_b = batch_var[j] * batch_size;
            double m2 = m_a + m_b + delta * delta * count_ * batch_size / total;
            mean_[j] = new_mean;
            var_[j] = m2 / total;
        }
        count_ = total;
    }

    void normalize(const float* in, float* out, int batch_size) const {
        for (int i = 0; i < batch_size; i++) {
            const float* src = in + i * size_;
            float* dst = out + i * size_;
            for (int j = 0; j < size_; j++) {
                float std_val = (float)std::sqrt(var_[j] + (double)eps_);
                float normed = ((float)(src[j] - (float)mean_[j])) / std_val;
                if (normed > clip_) normed = clip_;
                if (normed < -clip_) normed = -clip_;
                dst[j] = normed;
            }
        }
    }
};
