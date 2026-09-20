// =============================================================================
// tb_lfsr16_v2.v — race-free version.
// Stimulus is driven on the negedge (settled well before the sampling posedge);
// DUT output is sampled shortly after the posedge. Compares against the
// Python golden vector bit-exactly.
// =============================================================================
`timescale 1ns/1ps

module tb_lfsr16_v2;

    localparam N_VECTORS = 2000;

    reg clk = 0;
    reg rst_n = 0;
    reg advance = 0;
    wire [15:0] value;

    reg [15:0] golden [0:N_VECTORS-1];
    integer i;
    integer errors = 0;

    lfsr16 #(.SEED(16'hACE1)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .advance(advance),
        .value(value)
    );

    always #5 clk = ~clk;

    initial begin
        $readmemh("/home/claude/fpga_snn/sim/lfsr_golden.hex", golden);

        // Async reset, released on a negedge (safely before any sampled posedge)
        rst_n  = 0;
        advance = 0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        // value should now be exactly SEED (no advance has occurred yet)
        if (value !== 16'hACE1) begin
            $display("INIT MISMATCH: value=%h expected ACE1 (unadvanced seed)", value);
            errors = errors + 1;
        end

        for (i = 0; i < N_VECTORS; i = i + 1) begin
            advance = 1;          // driven on negedge, stable through next posedge
            @(posedge clk);       // DUT computes next() here
            #1;                   // let nonblocking update settle
            if (value !== golden[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH at step %0d: DUT=%h EXPECTED=%h", i, value, golden[i]);
            end
            @(negedge clk);
            advance = 0;           // deassert cleanly on negedge
        end

        if (errors == 0)
            $display("RESULT: PASS - all %0d LFSR values bit-exact match to Python reference (seed=0xACE1)", N_VECTORS);
        else
            $display("RESULT: FAIL - %0d / %0d mismatches", errors, N_VECTORS);

        $finish;
    end

endmodule
