// =============================================================================
// tb_bias_mem.v
// Confirms weight_mem.v (NUM_PRE=1 case) correctly serves bias vectors,
// against the REAL uploaded b1_hidden.mem / b2_output.mem files.
// =============================================================================
`timescale 1ns/1ps

module tb_bias_mem;

    reg clk = 0;
    always #5 clk = ~clk;

    reg  [$clog2(128)-1:0] addr_b1;
    wire signed [15:0] data_b1;

    weight_mem #(
        .NUM_PRE(1), .NUM_POST(128), .WIDTH(16),
        .MEM_FILE("/home/claude/fpga_snn/sim/b1_hidden.mem")
    ) b1_mem (
        .clk(clk), .addr(addr_b1), .data_out(data_b1)
    );

    reg  [$clog2(2)-1:0] addr_b2;
    wire signed [15:0] data_b2;

    weight_mem #(
        .NUM_PRE(1), .NUM_POST(2), .WIDTH(16),
        .MEM_FILE("/home/claude/fpga_snn/sim/b2_output.mem")
    ) b2_mem (
        .clk(clk), .addr(addr_b2), .data_out(data_b2)
    );

    integer errors = 0;

    task check_b1(input [31:0] a, input [15:0] expected);
        begin
            addr_b1 = a;
            @(posedge clk); #1;
            if (data_b1 !== expected) begin
                errors = errors + 1;
                $display("b1 MISMATCH addr=%0d got=%h expected=%h", a, data_b1, expected);
            end else
                $display("b1[%0d] OK (%h)", a, data_b1);
        end
    endtask

    task check_b2(input [31:0] a, input [15:0] expected);
        begin
            addr_b2 = a;
            @(posedge clk); #1;
            if (data_b2 !== expected) begin
                errors = errors + 1;
                $display("b2 MISMATCH addr=%0d got=%h expected=%h", a, data_b2, expected);
            end else
                $display("b2[%0d] OK (%h)", a, data_b2);
        end
    endtask

    initial begin
        addr_b1 = 0; addr_b2 = 0;
        @(posedge clk); #1;

        check_b1(0, 16'h4277);
        check_b1(1, 16'h1522);
        check_b1(2, 16'hdc9f);
        check_b1(3, 16'h16d3);
        check_b1(127, 16'h1f0c); // last element, independently recomputed from b1_raw.npy

        check_b2(0, 16'h7332);
        check_b2(1, 16'h8cce);

        if (errors == 0)
            $display("RESULT: PASS - bias memory verified against real b1/b2 files (first 4 + last of b1, both of b2, all bit-exact)");
        else
            $display("RESULT: FAIL - %0d mismatches", errors);

        $finish;
    end

endmodule
