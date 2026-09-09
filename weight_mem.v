// =============================================================================
// weight_mem.v
// -----------------------------------------------------------------------------
// Synapse weight memory for one layer (W1: 640x128, or W2: 128x2).
// Format verified bit-exact against the real files produced by
// export_weights_for_verilog() in train_snn.py:
//   - signed 16-bit two's complement, one hex value per line
//   - row-major: address = pre_idx * NUM_POST + post_idx
//   - a leading "// ..." comment line (Icarus/most $readmemh implementations
//     skip // comments, so the file loads directly with no preprocessing)
//
// One row (NUM_POST contiguous weights) is read per active presynaptic spike,
// starting at address = pre_idx * NUM_POST, for spike-gated accumulation
// (see mac_accumulator.v, not yet implemented).
// =============================================================================
module weight_mem #(
    parameter NUM_PRE  = 640,
    parameter NUM_POST = 128,
    parameter WIDTH    = 16,
    parameter MEM_FILE = ""              // path passed at instantiation/elaboration
) (
    input  wire                          clk,
    input  wire [$clog2(NUM_PRE*NUM_POST)-1:0] addr,   // pre_idx*NUM_POST + post_idx
    output reg  signed [WIDTH-1:0]       data_out
);

    reg signed [WIDTH-1:0] mem [0:(NUM_PRE*NUM_POST)-1];

    initial begin
        if (MEM_FILE != "")
            $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk) begin
        data_out <= mem[addr];
    end

endmodule
