// =============================================================================
// v_mem.v
// -----------------------------------------------------------------------------
// Simple dual-port RAM to hold membrane potentials.
// Starts at 0, readable and writable.
// =============================================================================
module v_mem #(
    parameter N_NEURONS = 128,
    parameter WIDTH = 32
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,   // synchronous clear/reset for all entries
    input  wire wr_en,
    input  wire [$clog2(N_NEURONS)-1:0] wr_addr,
    input  wire signed [WIDTH-1:0]   wr_data,
    input  wire [$clog2(N_NEURONS)-1:0] rd_addr,
    output reg  signed [WIDTH-1:0]   rd_data
);
    
    reg signed [WIDTH-1:0] mem [0:N_NEURONS-1];
    integer i;

    // We can't cleanly infer RAM if we zero it out sequentially in hardware 
    // without taking many cycles. But for FPGA BRAM, you often don't need a
    // single-cycle clear if the FSM can just spend 128 cycles clearing it, 
    // OR if we just clear upon start.
    // Wait, the project doesn't have a 128-cycle clear FSM state.
    // Instead, let's just initialize it to 0 and rely on the fact that
    // the FSM guarantees we write to it, or we can use an internal 'clear' logic.
    // Actually, typical BRAM inference requires NOT having a blanket clear.
    // For this prototype, I'll use a blanket clear; Quartus will just synthesize it 
    // as registers if it can't clear a BRAM in 1 cycle, which is only 128x32 = 4K FFs.
    // Since we saved 32,000 FFs by killing spike_storage, 4K is fine!
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<N_NEURONS; i=i+1) mem[i] <= 0;
            rd_data <= 0;
        end else begin
            if (start) begin
                for (i=0; i<N_NEURONS; i=i+1) mem[i] <= 0;
            end else if (wr_en) begin
                mem[wr_addr] <= wr_data;
            end
            
            // Read is synchronous
            rd_data <= mem[rd_addr];
        end
    end

endmodule
