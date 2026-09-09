// =============================================================================
// threshold_mem.v
// Unsigned 16-bit threshold ROM, one value per input neuron (row-major,
// freq*20+time), loaded from a .mem file generated with the exact formula
// from deterministic_rate_encode() in audio_to_spike.py:
//   thresholds_int = (clip(feature.flatten()*0.9, 0, 1) * 65535).astype(uint32)
// NOTE the truncating cast -- this is floor(), not round(). The Python
// generator (gen_rate_encoder_golden.py) reproduces this exactly.
// =============================================================================
module threshold_mem #(
    parameter N_NEURONS = 640,
    parameter WIDTH     = 16,
    parameter MEM_FILE  = ""
) (
    input  wire                          clk,
    input  wire [$clog2(N_NEURONS)-1:0]  addr,
    output reg  [WIDTH-1:0]              data_out
);

    reg [WIDTH-1:0] mem [0:N_NEURONS-1];

    initial begin
        if (MEM_FILE != "")
            $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk) begin
        data_out <= mem[addr];
    end

endmodule
