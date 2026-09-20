`timescale 1ns/1ps

module tb_top_8;
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
        
        // Sample 0
        key = 2'b11; 
        #200;
        wait(uut.u_snn.done == 1'b1);
        @(posedge clk);

        // Sample 1
        #5000; key = 2'b10;
        @(posedge uut.u_snn.start); wait(uut.u_snn.done == 1'b1); @(posedge clk);
        
        // Sample 2
        #5000; key = 2'b01; 
        @(posedge uut.u_snn.start); wait(uut.u_snn.done == 1'b1); @(posedge clk);

        // Sample 3
        #5000; key = 2'b00; 
        @(posedge uut.u_snn.start); wait(uut.u_snn.done == 1'b1); @(posedge clk);

        // Sample 0 (Repeat)
        #5000; key = 2'b11; 
        @(posedge uut.u_snn.start); wait(uut.u_snn.done == 1'b1); @(posedge clk);

        // Sample 1 (Repeat)
        #5000; key = 2'b10; 
        @(posedge uut.u_snn.start); wait(uut.u_snn.done == 1'b1); @(posedge clk);

        // Sample 2 (Repeat)
        #5000; key = 2'b01; 
        @(posedge uut.u_snn.start); wait(uut.u_snn.done == 1'b1); @(posedge clk);

        // Sample 3 (Repeat)
        #5000; key = 2'b00; 
        @(posedge uut.u_snn.start); wait(uut.u_snn.done == 1'b1); @(posedge clk);

        #10000;
        $finish;
    end
endmodule
