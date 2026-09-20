`timescale 1ns/1ps

module tb_snn_top;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    // ---- ambient instance ----
    reg start_a;
    wire done_a, winner_a;
    wire signed [63:0] z2_0_a, z2_1_a;

    snn_top #(
        .THRESH_MEM_FILE("/home/claude/fpga_snn/sim/threshold_ambient.mem"),
        .W1_MEM_FILE("/home/claude/fpga_snn/sim/weights/W1_folded_input_hidden.mem"),
        .B1_MEM_FILE("/home/claude/fpga_snn/sim/weights/b1_folded_hidden.mem"),
        .SIGMOID_MEM_FILE("/home/claude/fpga_snn/sim/sigmoid_lut.mem"),
        .W2_MEM_FILE("/home/claude/fpga_snn/sim/W2_hidden_output.mem"),
        .B2_MEM_FILE("/home/claude/fpga_snn/sim/b2_output.mem")
    ) dut_ambient (
        .clk(clk), .rst_n(rst_n), .start(start_a),
        .done(done_a), .winner(winner_a), .z2_total0(z2_0_a), .z2_total1(z2_1_a)
    );

    // ---- drone instance ----
    reg start_d;
    wire done_d, winner_d;
    wire signed [63:0] z2_0_d, z2_1_d;

    snn_top #(
        .THRESH_MEM_FILE("/home/claude/fpga_snn/sim/threshold_drone.mem"),
        .W1_MEM_FILE("/home/claude/fpga_snn/sim/weights/W1_folded_input_hidden.mem"),
        .B1_MEM_FILE("/home/claude/fpga_snn/sim/weights/b1_folded_hidden.mem"),
        .SIGMOID_MEM_FILE("/home/claude/fpga_snn/sim/sigmoid_lut.mem"),
        .W2_MEM_FILE("/home/claude/fpga_snn/sim/W2_hidden_output.mem"),
        .B2_MEM_FILE("/home/claude/fpga_snn/sim/b2_output.mem")
    ) dut_drone (
        .clk(clk), .rst_n(rst_n), .start(start_d),
        .done(done_d), .winner(winner_d), .z2_total0(z2_0_d), .z2_total1(z2_1_d)
    );

    reg signed [63:0] exp_z2_0, exp_z2_1;
    integer exp_winner, fd, code;

    task check_sample(input [1023:0] golden_file, input [1023:0] label,
                       input got_winner, input signed [63:0] got_z0, input signed [63:0] got_z1);
        begin
            fd = $fopen(golden_file, "r");
            code = $fscanf(fd, "%d", exp_z2_0);
            code = $fscanf(fd, "%d", exp_z2_1);
            code = $fscanf(fd, "%d", exp_winner);
            $fclose(fd);
            $display("[%0s] winner=%0b (expected %0d)  z2_0=%0d (exp %0d)  z2_1=%0d (exp %0d)",
                      label, got_winner, exp_winner, got_z0, exp_z2_0, got_z1, exp_z2_1);
            if (got_winner === exp_winner[0] && got_z0 === exp_z2_0 && got_z1 === exp_z2_1)
                $display("RESULT [%0s]: PASS - snn_top matches manually-verified golden exactly", label);
            else
                $display("RESULT [%0s]: FAIL", label);
        end
    endtask

    initial begin
        rst_n = 0; start_a = 0; start_d = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("=== snn_top: ambient ===");
        start_a = 1;
        @(negedge clk);
        start_a = 0;
        while (!done_a) @(negedge clk);
        check_sample("/home/claude/fpga_snn/sim/golden_output_folded_ambient.txt", "ambient", winner_a, z2_0_a, z2_1_a);

        $display("=== snn_top: drone ===");
        start_d = 1;
        @(negedge clk);
        start_d = 0;
        while (!done_d) @(negedge clk);
        check_sample("/home/claude/fpga_snn/sim/golden_output_folded_drone.txt", "drone", winner_d, z2_0_d, z2_1_d);

        $finish;
    end

endmodule
