#pragma once
#include <memory>
#include <vector>

class LogisticRegression {
    public:
        LogisticRegression();
        double train();
        // std::vector<double> predict();
};

double train();