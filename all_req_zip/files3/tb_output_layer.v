`timescale 1ns/1ps

module tb_output_layer;

    localparam N_NEURONS = 640;
    localparam N_POST    = 128;
    localparam N_STEPS   = 50;

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
        .W1_MEM_FILE("/home/claude/fpga_snn/sim/W1_input_hidden.mem"),
        .B1_MEM_FILE("/home/claude/fpga_snn/sim/b1_hidden.mem"),
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
    integer t, h;
    integer fd, code;
    reg signed [63:0] exp_z2_0, exp_z2_1;
    integer exp_winner;

    task run_sample(input [1023:0] spike_file, input [1023:0] golden_file, input [1023:0] label);
        begin
            $display("=== Running FULL PIPELINE for: %0s ===", label);
            $readmemh(spike_file, spike_mem);

            // ---- hidden layer: process all 50 timesteps ----
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

            // ---- output layer: stream all 128 (h, a1) pairs ----
            @(negedge clk);
            start_out = 1;
            @(negedge clk);
            start_out = 0;

            for (h = 0; h < N_POST; h = h + 1) begin
                rd_addr = h;
                repeat (5) @(posedge clk);  // let hidden_neuron's pipeline settle
                #1;
                a1_value = a1_out;
                h_index = h;
                @(negedge clk);
                h_valid = 1;
                @(negedge clk);
                h_valid = 0;
                while (busy_out) @(negedge clk);
            end

            // ---- finalize: add bias, argmax ----
            @(negedge clk);
            finalize = 1;
            @(negedge clk);
            finalize = 0;
            while (!done) @(negedge clk);
            @(posedge clk); #1;

            // ---- compare against golden ----
            fd = $fopen(golden_file, "r");
            code = $fscanf(fd, "%d", exp_z2_0);
            code = $fscanf(fd, "%d", exp_z2_1);
            code = $fscanf(fd, "%d", exp_winner);
            $fclose(fd);

            $display("  z2_total0=%0d (expected %0d)", z2_total0, exp_z2_0);
            $display("  z2_total1=%0d (expected %0d)", z2_total1, exp_z2_1);
            $display("  winner=%0d (expected %0d) [0=ambient, 1=drone]", winner, exp_winner);

            if (z2_total0 === exp_z2_0 && z2_total1 === exp_z2_1 && winner === exp_winner[0])
                $display("RESULT [%0s]: PASS - full pipeline bit-exact, classification=%0s",
                          label, winner ? "DRONE" : "AMBIENT");
            else
                $display("RESULT [%0s]: FAIL - mismatch", label);
        end
    endtask

    initial begin
        rst_n = 0; start_hidden = 0; spike_bus = 0; timestep_valid = 0; rd_addr = 0;
        start_out = 0; h_index = 0; a1_value = 0; h_valid = 0; finalize = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        run_sample("/home/claude/fpga_snn/sim/spike_bus_ambient.hex",
                   "/home/claude/fpga_snn/sim/golden_output_ambient.txt", "ambient");

        run_sample("/home/claude/fpga_snn/sim/spike_bus_drone.hex",
                   "/home/claude/fpga_snn/sim/golden_output_drone.txt", "drone");

        $finish;
    end

endmodule
