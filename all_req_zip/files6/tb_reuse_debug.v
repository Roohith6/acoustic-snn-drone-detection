`timescale 1ns/1ps

module tb_reuse_debug;

    localparam N_NEURONS = 640;
    localparam N_POST    = 128;
    localparam N_STEPS   = 50;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

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
    integer t, h, run;

    task run_once(input integer run_num);
        begin
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

            @(negedge clk);
            finalize = 1;
            @(negedge clk);
            finalize = 0;
            while (!done) @(negedge clk);
            @(posedge clk); #1;

            $display("RUN %0d: winner=%0b z0=%0d z1=%0d", run_num, winner, z2_total0, z2_total1);
        end
    endtask

    initial begin
        rst_n = 0; start_hidden = 0; spike_bus = 0; timestep_valid = 0; rd_addr = 0;
        start_out = 0; h_index = 0; a1_value = 0; h_valid = 0; finalize = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        $readmemh("/home/claude/fpga_snn/sim/spike_bus_ambient.hex", spike_mem);
        run_once(1);
        $display("  (expect ambient: z0=1852153522 z1=-1668125759 winner=0)");

        $readmemh("/home/claude/fpga_snn/sim/spike_bus_drone.hex", spike_mem);
        run_once(2);
        $display("  (expect drone: z0=-4728301355 z1=2842504525 winner=1)");

        $readmemh("/home/claude/fpga_snn/sim/spike_bus_ambient.hex", spike_mem);
        run_once(3);
        $display("  (expect ambient again: z0=1852153522 z1=-1668125759 winner=0)");

        $finish;
    end

endmodule
