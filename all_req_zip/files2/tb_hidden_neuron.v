`timescale 1ns/1ps

module tb_hidden_neuron;

    localparam N_NEURONS = 640;
    localparam N_POST    = 128;
    localparam N_STEPS   = 50;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    reg start;
    reg [N_NEURONS-1:0] spike_bus;
    reg timestep_valid;
    wire busy;
    reg [$clog2(N_POST)-1:0] rd_addr;
    wire [15:0] a1_out;

    hidden_neuron #(
        .N_NEURONS(N_NEURONS), .N_POST(N_POST),
        .W1_MEM_FILE("/home/claude/fpga_snn/sim/W1_input_hidden.mem"),
        .B1_MEM_FILE("/home/claude/fpga_snn/sim/b1_hidden.mem"),
        .SIGMOID_MEM_FILE("/home/claude/fpga_snn/sim/sigmoid_lut.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .spike_bus(spike_bus), .timestep_valid(timestep_valid), .busy(busy),
        .rd_addr(rd_addr), .a1_out(a1_out)
    );

    reg [N_NEURONS-1:0] spike_mem [0:N_STEPS-1];
    integer golden [0:N_POST-1];
    integer fd, code, val, t, h;
    integer errors;
    integer max_diff, d;

    task run_sample(input [1023:0] spike_file, input [1023:0] golden_file, input [1023:0] label);
        begin
            $display("=== Running sample: %0s ===", label);
            $readmemh(spike_file, spike_mem);

            fd = $fopen(golden_file, "r");
            h = 0;
            while (!$feof(fd) && h < N_POST) begin
                code = $fscanf(fd, "%d", val);
                golden[h] = val;
                h = h + 1;
            end
            $fclose(fd);

            @(negedge clk);
            start = 1;
            @(negedge clk);
            start = 0;

            for (t = 0; t < N_STEPS; t = t + 1) begin
                @(negedge clk);
                spike_bus = spike_mem[t];
                timestep_valid = 1;
                @(negedge clk);
                timestep_valid = 0;
                while (busy) @(negedge clk);
            end

            // read out all 128 hidden activations, 4-cycle pipeline latency per address
            errors = 0;
            max_diff = 0;
            for (h = 0; h < N_POST; h = h + 1) begin
                rd_addr = h;
                repeat (5) @(posedge clk);
                #1;
                d = a1_out - golden[h];
                if (d < 0) d = -d;
                if (d > max_diff) max_diff = d;
                if (a1_out !== golden[h]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("MISMATCH h=%0d got=%0d expected=%0d (diff=%0d)", h, a1_out, golden[h], d);
                end
            end

            if (errors == 0)
                $display("RESULT [%0s]: PASS - all %0d hidden activations bit-exact (max diff=%0d)", label, N_POST, max_diff);
            else
                $display("RESULT [%0s]: %0d/%0d mismatches, max diff=%0d", label, errors, N_POST, max_diff);
        end
    endtask

    initial begin
        rst_n = 0; start = 0; spike_bus = 0; timestep_valid = 0; rd_addr = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        run_sample("/home/claude/fpga_snn/sim/spike_bus_ambient.hex",
                   "/home/claude/fpga_snn/sim/golden_a1_ambient.txt", "ambient");

        run_sample("/home/claude/fpga_snn/sim/spike_bus_drone.hex",
                   "/home/claude/fpga_snn/sim/golden_a1_drone.txt", "drone");

        $finish;
    end

endmodule
