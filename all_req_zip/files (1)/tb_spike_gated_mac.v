`timescale 1ns/1ps

module tb_spike_gated_mac;

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
    reg [$clog2(N_POST)-1:0] acc_rd_addr;
    wire signed [31:0] acc_rd_data;

    spike_gated_mac #(
        .N_NEURONS(N_NEURONS), .N_POST(N_POST), .WEIGHT_WIDTH(16), .ACC_WIDTH(32),
        .WEIGHT_MEM_FILE("/home/claude/fpga_snn/sim/W1_input_hidden.mem")
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .spike_bus(spike_bus), .timestep_valid(timestep_valid), .busy(busy),
        .acc_rd_addr(acc_rd_addr), .acc_rd_data(acc_rd_data)
    );

    reg [N_NEURONS-1:0] spike_mem [0:N_STEPS-1];
    integer golden [0:N_POST-1];

    integer fd, code, val, t, h;
    integer errors;

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

            // reset accumulators
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
                // wait until not busy before feeding next timestep
                while (busy) @(negedge clk);
            end

            // read out and compare
            errors = 0;
            for (h = 0; h < N_POST; h = h + 1) begin
                acc_rd_addr = h;
                @(posedge clk); #1;
                @(posedge clk); #1; // registered read, 1-cycle latency
                if (acc_rd_data !== golden[h]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("MISMATCH h=%0d got=%0d expected=%0d", h, acc_rd_data, golden[h]);
                end
            end

            if (errors == 0)
                $display("RESULT [%0s]: PASS - all %0d hidden accumulators bit-exact", label, N_POST);
            else
                $display("RESULT [%0s]: FAIL - %0d/%0d mismatches", label, errors, N_POST);
        end
    endtask

    initial begin
        rst_n = 0; start = 0; spike_bus = 0; timestep_valid = 0; acc_rd_addr = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        run_sample("/home/claude/fpga_snn/sim/spike_bus_ambient.hex",
                   "/home/claude/fpga_snn/sim/golden_raw_acc_ambient.txt", "ambient");

        run_sample("/home/claude/fpga_snn/sim/spike_bus_drone.hex",
                   "/home/claude/fpga_snn/sim/golden_raw_acc_drone.txt", "drone");

        $finish;
    end

endmodule
