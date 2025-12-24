#pragma once

#include <cmath>

inline double wrap01(double x) {
    x -= std::floor(x);
    if (x < 0.0) {
        x += 1.0;
    }
    if (x >= 1.0) {
        x -= 1.0;
    }
    return x;
}

inline double wrap_deg(double deg) {
    deg = std::fmod(deg, 360.0);
    if (deg <= -180.0) {
        deg += 360.0;
    } else if (deg > 180.0) {
        deg -= 360.0;
    }
    return deg;
}

