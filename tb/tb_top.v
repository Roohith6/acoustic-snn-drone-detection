`timescale 1ns/1ps

module tb_top;

    reg clk;
    reg [1:0] key;
    wire [7:0] ledg;

    de2_top uut (
        .CLOCK_50(clk),
        .KEY(key),
        .LEDG(ledg)
    );

    always #10 clk = ~clk;

    wire [127:0] hid_spikes = uut.u_snn.u_hidden.out_spike_bus;

    initial begin
        clk = 0;
        key = 2'b11;    // sample_sel=00 after inversion
        #200;
        wait(uut.u_snn.done == 1'b1);
        @(posedge clk);
        $display("Inference 1 Done! Winner: %b, Ambient: %3d, Drone: %3d, t_exit: %2d",
            uut.u_snn.winner,
            uut.u_snn.spike_cnt_ambient,
            uut.u_snn.spike_cnt_drone,
            uut.u_snn.t_exit);

        #5000;
        key = 2'b10;
        @(posedge uut.u_snn.start);
        wait(uut.u_snn.done == 1'b1);
        @(posedge clk);
        $display("Inference 2 Done! Winner: %b, Ambient: %3d, Drone: %3d, t_exit: %2d",
            uut.u_snn.winner,
            uut.u_snn.spike_cnt_ambient,
            uut.u_snn.spike_cnt_drone,
            uut.u_snn.t_exit);
        $finish;
    end
endmodule
