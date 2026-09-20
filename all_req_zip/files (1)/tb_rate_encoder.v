`timescale 1ns/1ps

module tb_rate_encoder;

    localparam N_NEURONS = 640;
    localparam N_STEPS   = 50;
    localparam TOTAL     = N_NEURONS * N_STEPS;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    // ================= AMBIENT =================
    reg start_a = 0;
    wire spike_valid_a, spike_bit_a, done_a;
    wire [$clog2(N_STEPS)-1:0] out_t_a;
    wire [$clog2(N_NEURONS)-1:0] out_n_a;

    rate_encoder #(
        .N_NEURONS(N_NEURONS), .N_STEPS(N_STEPS), .WIDTH(16), .SEED(16'hACE1),
        .THRESH_MEM_FILE("/home/claude/fpga_snn/sim/threshold_ambient.mem")
    ) dut_ambient (
        .clk(clk), .rst_n(rst_n), .start(start_a),
        .spike_valid(spike_valid_a), .spike_bit(spike_bit_a),
        .out_t(out_t_a), .out_n(out_n_a), .done(done_a)
    );

    // ================= DRONE =================
    reg start_d = 0;
    wire spike_valid_d, spike_bit_d, done_d;
    wire [$clog2(N_STEPS)-1:0] out_t_d;
    wire [$clog2(N_NEURONS)-1:0] out_n_d;

    rate_encoder #(
        .N_NEURONS(N_NEURONS), .N_STEPS(N_STEPS), .WIDTH(16), .SEED(16'hACE1),
        .THRESH_MEM_FILE("/home/claude/fpga_snn/sim/threshold_drone.mem")
    ) dut_drone (
        .clk(clk), .rst_n(rst_n), .start(start_d),
        .spike_valid(spike_valid_d), .spike_bit(spike_bit_d),
        .out_t(out_t_d), .out_n(out_n_d), .done(done_d)
    );

    reg [0:0] golden_a [0:TOTAL-1];
    reg [0:0] golden_d [0:TOTAL-1];
    integer errors_a, received_a;
    integer errors_d, received_d;
    integer j;

    initial begin
        $readmemh("/home/claude/fpga_snn/sim/golden_spikes_ambient.hex", golden_a);
        $readmemh("/home/claude/fpga_snn/sim/golden_spikes_drone.hex", golden_d);

        rst_n = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // ---- run ambient ----
        errors_a = 0; received_a = 0;
        start_a = 1;
        @(negedge clk);
        start_a = 0;
        while (!done_a) begin
            @(posedge clk); #1;
            if (spike_valid_a) begin
                j = out_t_a * N_NEURONS + out_n_a;
                if (spike_bit_a !== golden_a[j][0]) begin
                    errors_a = errors_a + 1;
                    if (errors_a <= 10)
                        $display("AMBIENT MISMATCH idx=%0d (t=%0d,n=%0d) got=%b expected=%b",
                                  j, out_t_a, out_n_a, spike_bit_a, golden_a[j]);
                end
                received_a = received_a + 1;
            end
        end
        @(posedge clk); #1;
        if (errors_a == 0 && received_a == TOTAL)
            $display("RESULT [ambient]: PASS - all %0d bits bit-exact", TOTAL);
        else
            $display("RESULT [ambient]: FAIL - %0d errors, %0d/%0d bits received", errors_a, received_a, TOTAL);

        // ---- run drone (independent instance, runs after ambient; both share clk/rst_n) ----
        errors_d = 0; received_d = 0;
        @(negedge clk);              // align start pulse the same way the ambient run did
        start_d = 1;
        @(negedge clk);
        start_d = 0;
        while (!done_d) begin
            @(posedge clk); #1;
            if (spike_valid_d) begin
                j = out_t_d * N_NEURONS + out_n_d;
                if (spike_bit_d !== golden_d[j][0]) begin
                    errors_d = errors_d + 1;
                    if (errors_d <= 10)
                        $display("DRONE MISMATCH idx=%0d (t=%0d,n=%0d) got=%b expected=%b",
                                  j, out_t_d, out_n_d, spike_bit_d, golden_d[j]);
                end
                received_d = received_d + 1;
            end
        end
        @(posedge clk); #1;
        if (errors_d == 0 && received_d == TOTAL)
            $display("RESULT [drone]: PASS - all %0d bits bit-exact", TOTAL);
        else
            $display("RESULT [drone]: FAIL - %0d errors, %0d/%0d bits received", errors_d, received_d, TOTAL);

        $finish;
    end

endmodule
