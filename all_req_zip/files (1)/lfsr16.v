// =============================================================================
// lfsr16.v
// -----------------------------------------------------------------------------
// 16-bit Fibonacci LFSR, bit-exact match to the Python reference:
//
//   class LFSR16:
//       def __init__(self, seed=0xACE1):
//           self.state = seed & 0xFFFF
//       def next(self):
//           s = self.state
//           bit = ((s>>15)^(s>>13)^(s>>12)^(s>>10)) & 1
//           self.state = ((s<<1)|bit) & 0xFFFF
//           return self.state
//
// Taps (0-indexed from LSB=bit0): 15, 13, 12, 10  (maximal-length, period 65535
// for any nonzero seed). Default seed 0xACE1, per frozen Encoding V1 contract.
//
// Semantics: state updates and the OUTPUT value are the POST-shift state,
// matching Python's next() which returns self.state AFTER the update.
// A synchronous "advance" pulse produces exactly one Python next() call.
// =============================================================================
module lfsr16 #(
    parameter [15:0] SEED = 16'hACE1
) (
    input  wire        clk,
    input  wire        rst_n,     // async active-low reset
    input  wire         advance,   // one Python next() call per cycle this is high
    output wire [15:0] value      // current registered state (post-update)
);

    reg [15:0] state;

    wire fb = state[15] ^ state[13] ^ state[12] ^ state[10];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Match Python __init__: seed&0xFFFF, and (seed==0 -> 0xACE1).
            // SEED parameter is expected to already be nonzero (0xACE1 default).
            state <= SEED;
        end else if (advance) begin
            state <= {state[14:0], fb};
        end
    end

    assign value = state;

endmodule
