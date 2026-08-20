#pragma once
#include <cstdint> // for integer types like uint32_t, uint_64_t
#include <array>
#include <utility> // we will be using std::pair for returning (x,y) pairs

class SobolGenerator
{
    public:
        using Point2D = std::pair<float, float> ;// creating an alias for (x(float),y(float)) type
        SobolGenerator(); // constructor
        Point2D next(); // increments teh internal state by one step and returns the next (x,y)
        void reset(); // resets teh generator back to index 0
        uint32_t currentIndex() const; // this returns how many points have been generated since the last reset
    private:
        static constexpr int BIT_WIDTH = 32; // we are taking 32 bits precision fr a dimension
        std::array<uint32_t, BIT_WIDTH> directionNumbersX;// fr storeing  32 predefined sobol direction no's used to generate the x coordinates
        std::array<uint32_t, BIT_WIDTH> directionNumbersY;// fr storeing  32 predefined sobol direction no's used to generate the x coordinates
        uint32_t stateX;// fr storing the current raw 32-bit sobol state for the x coordinate
        uint32_t stateY;// fr storing the current raw 32-bit sobol state for the x coordinate
        uint32_t index;// fr storing the current position in teh sobol sequence and to determine which direction no will be used next
        static int rightmostZeroBit(uint32_t k); //finds the position of the rightmost zero bit  so next() knows which direction no to XOR into the state

}
