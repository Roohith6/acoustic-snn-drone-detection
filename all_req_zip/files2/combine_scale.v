// =============================================================================
// combine_scale.v
// -----------------------------------------------------------------------------
// spike_gated_mac's raw_acc is a plain integer sum of quantized W1 weights
// (units: W1_float * W1_SCALE, summed over active spikes -- NOT yet divided
// by T=50). b1_int is quantized with its OWN, DIFFERENT scale factor
// (b1_SCALE != W1_SCALE -- confirmed from the real shipped .mem headers:
// W1 scale=5515.9670, b1 scale=3814.7672). Both need rescaling into a
// common base -- Q9.7, matching sigmoid_lut.v's input format -- before
// they can be added together as z1 = raw_acc/(T*W1_SCALE) + b1_int/b1_SCALE.
//
// Implemented as two fixed-point multiplies (by precomputed 30-bit
// constants A_FIXED/B_FIXED, see python_ref for derivation) rather than
// runtime division -- standard practice, maps to DSP multiplier(s) on
// synthesis.
//
// ACCURACY NOTE (measured against real data, not assumed): this rescale
// introduces up to ~2 LSB of Q9.7 rounding vs. computing z1 directly in
// float and quantizing once -- confirmed by comparing against the
// already-verified z1_q values from the sigmoid_lut step. Translates to
// at most ~0.004 error in the final sigmoid probability near z=0 (the
// steepest part of the curve), and effectively zero error anywhere
// already saturated. Accepted as a documented tradeoff, not silently
// ignored -- see PROJECT_REPORT.md.
// =============================================================================
module combine_scale #(
    parameter ACC_WIDTH = 32,
    parameter BIAS_WIDTH = 16,
    parameter signed [31:0] A_FIXED = 32'd498331,     // 2^7 / (50 * 5515.9670), Q?.30
    parameter signed [31:0] B_FIXED = 32'd36028137,   // 2^7 / 3814.7672,        Q?.30
    parameter SHIFT = 30
) (
    input  wire clk,
    input  wire signed [ACC_WIDTH-1:0]  raw_acc,
    input  wire signed [BIAS_WIDTH-1:0] b1_int,
    output reg  signed [15:0]           z1_q   // Q9.7, saturated to 16-bit signed
);

    wire signed [63:0] prodA = raw_acc * A_FIXED;
    wire signed [63:0] prodB = b1_int  * B_FIXED;

    wire signed [63:0] termA = prodA >>> SHIFT;
    wire signed [63:0] termB = prodB >>> SHIFT;
    wire signed [64:0] sum   = termA + termB;

    localparam signed [64:0] SAT_MAX = 65'sd32767;
    localparam signed [64:0] SAT_MIN = -65'sd32768;

    always @(posedge clk) begin
        if (sum > SAT_MAX)
            z1_q <= 16'sd32767;
        else if (sum < SAT_MIN)
            z1_q <= -16'sd32768;
        else
            z1_q <= sum[15:0];
    end

endmodule
