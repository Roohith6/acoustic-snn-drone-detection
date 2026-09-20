// =============================================================================
// tb_weight_mem.v
// Loads the ACTUAL W1_input_hidden.mem and W2_hidden_output.mem files you
// uploaded (produced by your real train_snn.py run) and checks specific
// addresses against values independently recomputed in Python from the
// raw .npy floats + the documented quantization formula. This proves
// weight_mem.v's addressing and $readmemh handling (including the
// leading "// ..." comment line) work against your real files, not a
// hand-crafted test file.
// =============================================================================
`timescale 1ns/1ps

module tb_weight_mem;

    reg clk = 0;
    always #5 clk = ~clk;

    // ---------------- W1: 640 x 128 ----------------
    reg  [$clog2(640*128)-1:0] addr1;
    wire signed [15:0] data1;

    weight_mem #(
        .NUM_PRE(640), .NUM_POST(128), .WIDTH(16),
        .MEM_FILE("/home/claude/fpga_snn/sim/W1_input_hidden.mem")
    ) w1_mem (
        .clk(clk), .addr(addr1), .data_out(data1)
    );

    // ---------------- W2: 128 x 2 ----------------
    reg  [$clog2(128*2)-1:0] addr2;
    wire signed [15:0] data2;

    weight_mem #(
        .NUM_PRE(128), .NUM_POST(2), .WIDTH(16),
        .MEM_FILE("/home/claude/fpga_snn/sim/W2_hidden_output.mem")
    ) w2_mem (
        .clk(clk), .addr(addr2), .data_out(data2)
    );

    integer errors = 0;

    task check_w1(input [31:0] a, input [15:0] expected);
        begin
            addr1 = a;
            @(posedge clk); #1;
            if (data1 !== expected) begin
                errors = errors + 1;
                $display("W1 MISMATCH addr=%0d got=%h expected=%h", a, data1, expected);
            end else
                $display("W1 addr=%0d OK (%h)", a, data1);
        end
    endtask

    task check_w2(input [31:0] a, input [15:0] expected);
        begin
            addr2 = a;
            @(posedge clk); #1;
            if (data2 !== expected) begin
                errors = errors + 1;
                $display("W2 MISMATCH addr=%0d got=%h expected=%h", a, data2, expected);
            end else
                $display("W2 addr=%0d OK (%h)", a, data2);
        end
    endtask

    initial begin
        addr1 = 0; addr2 = 0;
        @(posedge clk); #1;

        // First row values, already confirmed against the raw .mem file text directly
        check_w1(0, 16'h0d4c);
        check_w1(1, 16'hc009);
        check_w1(2, 16'h0644);
        check_w1(3, 16'hf2e3);

        // Independently recomputed spot checks, arbitrary interior/edge addresses
        check_w1(391,   16'hf3b0);   // pre=3,  post=7
        check_w1(81919, 16'hf65b);   // pre=639, post=127 (last element)

        check_w2(0,   16'h2be7);     // pre=0, post=0 (first line of file)
        check_w2(1,   16'he175);     // pre=0, post=1
        check_w2(255, 16'he762);     // pre=127, post=1 (last element)
        check_w2(100, 16'h1ded);     // pre=50, post=0

        if (errors == 0)
            $display("RESULT: PASS - weight_mem.v matches real uploaded .mem files exactly at all spot-checked addresses");
        else
            $display("RESULT: FAIL - %0d mismatches", errors);

        $finish;
    end

endmodule
