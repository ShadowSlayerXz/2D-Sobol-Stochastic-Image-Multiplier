#include "sobol_generator.hpp"

SobolGenerator::SobolGenerator() {
    index = 0;
    stateX = 0;
    stateY = 0;


    for (int i = 1; i <= BIT_WIDTH; i++) {
        directionNumbersX[i - 1] = static_cast<uint32_t>(1u) << (BIT_WIDTH - i);
    }

    for (int i = 1; i <= BIT_WIDTH; i++) {
        uint32_t base = static_cast<uint32_t>(1u) << (BIT_WIDTH - i);
        directionNumbersY[i - 1] = base ^ (base >> 1);
    }
}

int SobolGenerator::rightmostZeroBit(uint32_t k) {
    int position = 1;

    while (k & 1u) {
        k >>= 1;
        position++;
    }
    return position;
}

SobolGenerator::Point2D SobolGenerator::next() {
    int b = rightmostZeroBit(index);

    stateX ^= directionNumbersX[b - 1];
    stateY ^= directionNumbersY[b - 1];
    index++;
    float x = static_cast<float>(static_cast<double>(stateX) / 4294967296.0); // 2 pow32
    float y = static_cast<float>(static_cast<double>(stateY) / 4294967296.0);

    return Point2D(x, y);
}

void SobolGenerator::reset() {
    index = 0;
    stateX = 0;
    stateY = 0;
}
uint32_t SobolGenerator::currentIndex() const {
    return index;
}
