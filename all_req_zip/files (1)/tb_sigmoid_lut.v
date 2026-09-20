`timescale 1ns/1ps

module tb_sigmoid_lut;

    localparam N_VECTORS = 256;

    reg clk = 0;
    always #5 clk = ~clk;

    reg signed [15:0] z1_q;
    wire [15:0] sigmoid_out;

    sigmoid_lut #(
        .FRAC_BITS(7), .TABLE_SIZE(2048),
        .MEM_FILE("/home/claude/fpga_snn/sim/sigmoid_lut.mem")
    ) dut (
        .clk(clk), .z1_q(z1_q), .sigmoid_out(sigmoid_out)
    );

    reg [15:0] test_in  [0:N_VECTORS-1];
    reg [15:0] test_out [0:N_VECTORS-1];

    integer i;
    integer errors = 0;
    integer max_abs_diff = 0;
    integer diff;
    integer fd, code, zq_val, out_val;
    reg [1023:0] dummy_line;

    initial begin
        // parse "z1_q_hex expected_hex" pairs from the two-column hex file
        fd = $fopen("/home/claude/fpga_snn/sim/sigmoid_test_vectors.hex", "r");
        code = $fgets(dummy_line, fd); // skip the leading "// ..." comment line
        i = 0;
        while (!$feof(fd) && i < N_VECTORS) begin
            code = $fscanf(fd, "%h %h", zq_val, out_val);
            if (code == 2) begin
                test_in[i]  = zq_val[15:0];
                test_out[i] = out_val[15:0];
                i = i + 1;
            end
        end
        $fclose(fd);
        $display("Loaded %0d test vectors", i);

        // stream inputs, one per cycle; outputs valid 2 cycles later
        for (i = 0; i < N_VECTORS; i = i + 1) begin
            z1_q = test_in[i];
            @(posedge clk);
        end
        z1_q = 0;
        @(posedge clk); @(posedge clk); @(posedge clk); // drain pipeline

        $finish;
    end

    // separate always block samples sigmoid_out each cycle, correlating with a
    // 2-cycle-delayed index counter
    integer out_idx = -1; // corrected: valid sigmoid_out for test_in[0] appears 2 cycles after
                           // it's presented; this starting offset aligns the check correctly
                           // (found via simulation mismatch, not assumed correct up front)
    always @(posedge clk) begin
        #1;
        if (out_idx >= 0 && out_idx < N_VECTORS) begin
            diff = sigmoid_out - test_out[out_idx];
            if (diff < 0) diff = -diff;
            if (diff > max_abs_diff) max_abs_diff = diff;
            if (sigmoid_out !== test_out[out_idx]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH idx=%0d got=%h expected=%h (diff=%0d)",
                              out_idx, sigmoid_out, test_out[out_idx], diff);
            end
        end
        out_idx = out_idx + 1;
        if (out_idx == N_VECTORS) begin
            if (errors == 0)
                $display("RESULT: PASS - all %0d real hidden-neuron sigmoid values bit-exact (max diff=%0d)", N_VECTORS, max_abs_diff);
            else
                $display("RESULT: FAIL - %0d/%0d mismatches (max diff=%0d)", errors, N_VECTORS, max_abs_diff);
        end
    end

endmodule
