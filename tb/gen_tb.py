tb = '''\	imescale 1ns/1ps

module tb_top();
    reg clk;
    reg [1:0] key;
    wire [1:0] ledg;

    de2_top uut (
        .CLOCK_50(clk),
        .KEY(key),
        .LEDG(ledg)
    );

    always #10 clk = ~clk;

    initial begin
        clk = 0;
        key = 2'b11; // sample_sel = 00

        // Wait for power-on reset and first inference
        #200;
        
        wait(uut.u_snn.done == 1'b1);
        \("Inference 1 Done! Winner: %b, Ambient: %d, Drone: %d, t_exit: %d", 
            uut.u_snn.winner, uut.u_snn.spike_cnt_ambient, uut.u_snn.spike_cnt_drone, uut.u_snn.t_exit);

        // Try another key code
        #1000;
        key = 2'b10; // sample_sel = 01 (drone)
        wait(uut.u_snn.done == 1'b1);
        \("Inference 2 Done! Winner: %b, Ambient: %d, Drone: %d, t_exit: %d", 
            uut.u_snn.winner, uut.u_snn.spike_cnt_ambient, uut.u_snn.spike_cnt_drone, uut.u_snn.t_exit);

        #1000;
        \;
    end

    integer total_hidden_spikes = 0;
    integer i;
    always @(posedge clk) begin
        if (uut.u_snn.u_hidden.out_valid) begin
            for (i=0; i<128; i=i+1) begin
                if (uut.u_snn.u_hidden.out_spike_bus[i]) total_hidden_spikes = total_hidden_spikes + 1;
            end
            \("Time %t, TS %d, total hidden spikes so far: %d", \, uut.u_snn.t_cnt, total_hidden_spikes);
        end
    end
endmodule
'''

tb = tb.replace('\\', '').replace('\\$', '$')
with open('tb_top.v', 'w') as f:
    f.write(tb)
