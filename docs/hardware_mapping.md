# Hardware mapping of the 2D Sobol stochastic multiplier

> **Status: design sketch + analytic estimates. Nothing in this file was synthesised, simulated in RTL
> or measured.** Every gate count below is an *estimate* from the stated per-cell assumptions. The C++
> model in `src/` is the only executable reference (its numbers are in `README.md`).

## 1. Dataflow (one multiplier, one pixel channel)

```
            +-----------------+
 clk ------>|  SOBOL_INDEX    |  n = 0 .. N-1                (log2 N bits, wraps at N)
            +--------+--------+
                     | c = position of rightmost 0 of n   (priority encoder, log2 N inputs)
                     v
            +-----------------+      +----------------+
            |   DIR ROM x/y   |----->| SOBOL_X, SOBOL_Y|  state ^= DIR[c]   (XOR, one per dimension)
            +-----------------+      +--------+-------+
                                              | ref_x            | ref_y
        pixel A (8b) --------> [ CMP: ref_x < A ] --- a_k ---+
                                                              +--> AND --> [ ONES_COUNT += ] --> result
        pixel B (8b) --------> [ CMP: ref_y < B ] --- b_k ---+
                       BIT_COUNTER (0..N-1) stops the run when it reaches N
```

The Sobol generator, index counter and ROM are **shared** by every multiplier lane working on the same
(A, B) stream pair, exactly as `buildBank()` shares one reference sequence per operand in the software
model. Only the two comparators, the AND and the counter are per lane.

## 2. Registers

| Register | Width | Purpose |
|---|---|---|
| `SOBOL_INDEX` | log2(N) (8 for N=256, 10 for N=1024) | point number n; its Gray-code update picks the direction number |
| `SOBOL_X`, `SOBOL_Y` | software uses 32 b; hardware needs only as many bits as the comparators use (>= 8, 10 keeps the N=1024 net exact) | running Sobol state, `state ^= DIR[c]` |
| `DIR_X`, `DIR_Y` ROM | 2 x log2(N) words x state width | Joe-Kuo direction numbers for dimensions 1 and 2 (constants; with digital shift the shift value is one extra register per coordinate, XORed on output) |
| `BIT_COUNTER` | log2(N) + 1 | cycles elapsed in the current stream; `done` when it equals N |
| `ONES_COUNT` | log2(N) + 1 (9 b for N=256) | accumulates `a_k & b_k`; result = `round(255 * ONES_COUNT / N)` |
| `A_REG`, `B_REG` | 8 each | latched pixel values (per lane) |

`SOBOL_INDEX` and `BIT_COUNTER` count the same quantity and can be merged in a real design.

## 3. Control FSM

```
IDLE --start--> LOAD   : latch A,B; SOBOL_INDEX=0, SOBOL_X/Y = (0,0) (+shift), ONES_COUNT=0
LOAD ----------> RUN   : each cycle: bit = (ref_x<A)&(ref_y<B); ONES_COUNT += bit;
                          advance SOBOL_INDEX, state ^= DIR[c]
RUN --index==N-1--> DONE : result register valid, `done` asserted
DONE --ack--> IDLE
```

Latency is N cycles per product (N = 128 / 256 / 1024 in the software study), against one combinational
pass for a binary multiplier. N lanes of streams can instead be processed per cycle (the software AVX2
kernel does 256 stream bits per instruction); that trades area back for latency.

## 4. Gate-count estimate (labelled estimate, NAND2-equivalents)

Cell-cost assumptions (typical textbook values, **not** from a specific standard-cell library):
AND2 = 1.5 GE, XOR2 = 2.5 GE, half adder = 4 GE, full adder = 7 GE, D flip-flop = 5.5 GE,
n-bit magnitude comparator ~ 3.75 GE/bit.

**Reference: 8x8 unsigned array multiplier (combinational, 1 cycle)**

| Part | Count | GE |
|---|---|---|
| partial-product AND2 | 64 | 96 |
| full adders | 48 | 336 |
| half adders | 8 | 32 |
| **total** | | **~464 GE** |

**Stochastic lane, N = 256 (per multiplier lane, shared generator excluded)**

| Part | GE |
|---|---|
| 2 x 8-bit comparator (ref < pixel) | 2 x 30 = 60 |
| AND2 | 1.5 |
| ONES_COUNT 9-bit up-counter (9 DFF + incrementer) | 9 x 5.5 + 9 x 4 = ~86 |
| **total per lane** | **~148 GE** |

**Shared generator (amortised over all lanes using the same stream pair):** 8-bit index counter (~50 GE),
priority encoder (~15 GE), two 8-bit state registers + XOR update (~2 x 8 x 8 = ~128 GE),
DIR ROM (2 x 8 x 8 bits, ~60 GE as logic) -> roughly **250 GE total, shared**.

**What the estimate says, and what it does not**

* Per lane the stochastic datapath is roughly 1/3 of the array multiplier's area (~148 vs ~464 GE,
  ~68 % fewer GE) *under these assumptions*, before adding the shared ~250 GE and control. With a
  single lane the shared generator makes the stochastic design **larger** than the array multiplier;
  it only wins once about two or more lanes share one generator, and then only in area, not time.
* It pays for that with N = 128-1024 cycles per product, so energy-delay is not obviously better. No
  energy numbers are given because none were measured.
* The ">98 % gate savings" figure in the original project targets is **not supported** by this estimate
  and must not be claimed.
* The software model compares a 32-bit reference against `level/255` exactly. A hardware comparator on
  8 truncated bits compares against `level/256` instead; that small scaling difference is not modelled
  in the C++ code and its effect on MAE was not measured.

## 5. What would be needed to turn this into measurements

1. Write RTL (Verilog/VHDL) for the datapath above and a testbench that replays the C++ vectors.
2. Verify bit-exact agreement with `stochasticLevel()` for N = 128, 256, 1024.
3. Synthesise with a real cell library (or an FPGA flow) and report area/timing/power from the tool.
