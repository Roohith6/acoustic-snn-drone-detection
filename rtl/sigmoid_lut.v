// =============================================================================
// sigmoid_lut.v
// -----------------------------------------------------------------------------
// Fixed-point sigmoid: 1/(1+exp(-z)), for the hybrid hardware neuron
// (z1 = spike-gated accumulate of W1 + b1, see PROJECT_REPORT.md Session 3
// design decision -- this LUT is the "exact trained nonlinearity" part of
// that plan, not an invented threshold).
//
// FORMATS (chosen from real z1 data, see python_ref -- ambient samples stay
// within roughly +-10, drone samples reach +-58; sigmoid is already fully
// saturated well before +-16 either way, so precision is only needed near 0):
//   Input  z1_q : signed  Q9.7  (16-bit, range +-256.0, step 1/128 = 0.0078125)
//   Output      : unsigned Q0.16 (16-bit, range [0, 1), i.e. 65535 = ~1.0)
//
// Only the [0, 16.0) half of the curve is tabulated (2048 entries, step
// 1/128) -- the negative half is derived from the identity
// sigmoid(-x) = 1 - sigmoid(x), halving table size/BRAM vs. a full-range
// table for a symmetric function. Beyond |z1| >= 16.0 the result saturates
// to 0 or 65535 directly, no table access needed out there.
// =============================================================================
module sigmoid_lut #(
    parameter FRAC_BITS  = 7,
    parameter TABLE_SIZE = 2048,          // = 16.0 * 2^FRAC_BITS
    parameter MEM_FILE   = ""
) (
    input  wire                clk,
    input  wire signed [15:0]  z1_q,      // Q9.7 signed input
    output reg  [15:0]         sigmoid_out // Q0.16 unsigned output
);

    reg [15:0] table_mem [0:TABLE_SIZE-1];

    initial begin
        if (MEM_FILE != "")
            $readmemh(MEM_FILE, table_mem);
    end

    wire        sign_neg = z1_q[15];
    wire [15:0] mag      = sign_neg ? (~z1_q + 16'd1) : z1_q;  // abs(z1_q), 16-bit
    wire        saturate = (mag >= TABLE_SIZE[15:0]);

    reg  [15:0] table_val;
    reg         sign_neg_d;
    reg         saturate_d;

    always @(posedge clk) begin
        // registered ROM read, 1-cycle latency (matches weight_mem/threshold_mem style)
        table_val  <= table_mem[mag[$clog2(TABLE_SIZE)-1:0]];
        sign_neg_d <= sign_neg;
        saturate_d <= saturate;
    end

    always @(posedge clk) begin
        if (saturate_d)
            sigmoid_out <= sign_neg_d ? 16'h0000 : 16'hFFFF;
        else
            sigmoid_out <= sign_neg_d ? (16'hFFFF - table_val) : table_val;
    end

endmodule
