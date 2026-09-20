`timescale 1ns/1ps

module tb_batch_regression;

    localparam N_NEURONS = 640;
    localparam N_POST    = 128;
    localparam N_STEPS   = 50;
    localparam N_SAMPLES = 48;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    // ---- hidden layer ----
    reg start_hidden;
    reg [N_NEURONS-1:0] spike_bus;
    reg timestep_valid;
    wire busy_hidden;
    reg [$clog2(N_POST)-1:0] rd_addr;
    wire [15:0] a1_out;

    hidden_neuron #(
        .N_NEURONS(N_NEURONS), .N_POST(N_POST),
        .W1_MEM_FILE("/home/claude/fpga_snn/sim/weights/W1_folded_input_hidden.mem"),
        .B1_MEM_FILE("/home/claude/fpga_snn/sim/weights/b1_folded_hidden.mem"),
        .SIGMOID_MEM_FILE("/home/claude/fpga_snn/sim/sigmoid_lut.mem")
    ) u_hidden (
        .clk(clk), .rst_n(rst_n),
        .start(start_hidden), .spike_bus(spike_bus), .timestep_valid(timestep_valid), .busy(busy_hidden),
        .rd_addr(rd_addr), .a1_out(a1_out)
    );

    // ---- output layer ----
    reg start_out;
    reg [$clog2(N_POST)-1:0] h_index;
    reg [15:0] a1_value;
    reg h_valid;
    wire busy_out;
    reg finalize;
    wire done;
    wire winner;
    wire signed [63:0] z2_total0, z2_total1;

    output_layer #(
        .N_POST(N_POST), .N_OUT(2),
        .W2_MEM_FILE("/home/claude/fpga_snn/sim/W2_hidden_output.mem"),
        .B2_MEM_FILE("/home/claude/fpga_snn/sim/b2_output.mem")
    ) u_output (
        .clk(clk), .rst_n(rst_n),
        .start(start_out), .h_index(h_index), .a1_value(a1_value), .h_valid(h_valid), .busy(busy_out),
        .finalize(finalize), .done(done), .winner(winner),
        .z2_total0(z2_total0), .z2_total1(z2_total1)
    );

    reg [N_NEURONS-1:0] spike_mem [0:N_STEPS-1];
    integer t, h, s;
    integer n_samples_read;
    integer sample_idx [0:N_SAMPLES-1];
    integer true_label [0:N_SAMPLES-1];
    integer exp_winner [0:N_SAMPLES-1];
    reg signed [63:0] exp_z0 [0:N_SAMPLES-1];
    reg signed [63:0] exp_z1 [0:N_SAMPLES-1];

    integer fd, code;
    integer pass_count, fail_count;
    integer acc_correct;
    reg [1023:0] fname;
    reg [8*8-1:0] idx_str;

    initial begin
        // ---- load manifest ----
        fd = $fopen("/home/claude/fpga_snn/sim/batch_manifest.txt", "r");
        code = $fscanf(fd, "%d", n_samples_read);
        for (s = 0; s < n_samples_read; s = s + 1) begin
            code = $fscanf(fd, "%d %d %d %d %d", sample_idx[s], true_label[s], exp_winner[s], exp_z0[s], exp_z1[s]);
        end
        $fclose(fd);
        $display("Loaded manifest: %0d samples", n_samples_read);

        rst_n = 0; start_hidden = 0; spike_bus = 0; timestep_valid = 0; rd_addr = 0;
        start_out = 0; h_index = 0; a1_value = 0; h_valid = 0; finalize = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        pass_count = 0;
        fail_count = 0;
        acc_correct = 0;

        for (s = 0; s < n_samples_read; s = s + 1) begin
            $sformat(fname, "/home/claude/fpga_snn/sim/batch_spikes/sample_%03d.hex", sample_idx[s]);
            $readmemh(fname, spike_mem);

            // ---- hidden layer: 50 timesteps ----
            @(negedge clk);
            start_hidden = 1;
            @(negedge clk);
            start_hidden = 0;

            for (t = 0; t < N_STEPS; t = t + 1) begin
                @(negedge clk);
                spike_bus = spike_mem[t];
                timestep_valid = 1;
                @(negedge clk);
                timestep_valid = 0;
                while (busy_hidden) @(negedge clk);
            end

            // ---- output layer: 128 (h,a1) pairs ----
            @(negedge clk);
            start_out = 1;
            @(negedge clk);
            start_out = 0;

            for (h = 0; h < N_POST; h = h + 1) begin
                rd_addr = h;
                repeat (5) @(posedge clk);
                #1;
                a1_value = a1_out;
                h_index = h;
                @(negedge clk);
                h_valid = 1;
                @(negedge clk);
                h_valid = 0;
                while (busy_out) @(negedge clk);
            end

            // ---- finalize ----
            @(negedge clk);
            finalize = 1;
            @(negedge clk);
            finalize = 0;
            while (!done) @(negedge clk);
            @(posedge clk); #1;

            // ---- check ----
            if (z2_total0 === exp_z0[s] && z2_total1 === exp_z1[s] && winner === exp_winner[s][0]) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("SAMPLE %0d MISMATCH: got winner=%0b z0=%0d z1=%0d | expected winner=%0d z0=%0d z1=%0d",
                          sample_idx[s], winner, z2_total0, z2_total1, exp_winner[s], exp_z0[s], exp_z1[s]);
            end

            if (winner === true_label[s][0])
                acc_correct = acc_correct + 1;

            $display("sample %0d/%0d (idx=%0d, true=%0d): winner=%0b  [%0s]",
                      s+1, n_samples_read, sample_idx[s], true_label[s], winner,
                      (winner===true_label[s][0]) ? "CORRECT" : "wrong");
        end

        $display("\n=========================================");
        $display("RTL BATCH REGRESSION: %0d/%0d bit-exact match to golden", pass_count, n_samples_read);
        $display("RTL classification accuracy vs ground truth: %0d/%0d", acc_correct, n_samples_read);
        $display("=========================================");

        $finish;
    end

endmodule
