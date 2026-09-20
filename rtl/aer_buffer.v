// =============================================================================
// aer_buffer.v
// -----------------------------------------------------------------------------
// Replaces the 32,000-FF dense spike_storage with a sparse event FIFO.
// Writes events {t, n} during ENCODE.
// Reconstructs a 640-bit spike bus for a single timestep on demand for HIDDEN.
// =============================================================================
module aer_buffer #(
    parameter MAX_EVENTS = 4096,
    parameter N_NEURONS  = 640,
    parameter N_STEPS    = 50
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,             // clear pointers

    // Write side (from rate_encoder)
    input  wire       wr_en,
    input  wire [5:0] wr_t,
    input  wire [9:0] wr_n,

    // Read side (from snn_top)
    input  wire       fetch,       // pulse to begin fetching for query_t
    input  wire [5:0] query_t,
    
    output reg  [N_NEURONS-1:0] spike_bus_out,
    output reg        valid        // high when spike_bus_out is ready
);

    reg [15:0] mem [0:MAX_EVENTS-1];
    reg [$clog2(MAX_EVENTS):0] wr_ptr;
    reg [$clog2(MAX_EVENTS):0] rd_ptr;

    // FSM for assembling the bus
    reg [1:0] state;
    localparam S_IDLE = 2'd0;
    localparam S_READ = 2'd1;
    localparam S_WAIT = 2'd2; // BRAM read latency

    reg [$clog2(MAX_EVENTS):0] cur_rd;
    wire [15:0] mem_dout;
    
    // Simple read port (inferred BRAM)
    reg [15:0] read_reg;
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_ptr] <= {wr_t, wr_n};
        end
        read_reg <= mem[cur_rd];
    end
    assign mem_dout = read_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            state <= S_IDLE;
            spike_bus_out <= 0;
            valid <= 1'b0;
            cur_rd <= 0;
        end else begin
            if (start) begin
                wr_ptr <= 0;
                rd_ptr <= 0;
                state <= S_IDLE;
                valid <= 1'b0;
            end else if (wr_en) begin
                wr_ptr <= wr_ptr + 1;
            end
            
            case (state)
                S_IDLE: begin
                    if (fetch) begin
                        spike_bus_out <= 0;
                        valid <= 1'b0;
                        cur_rd <= rd_ptr;
                        if (rd_ptr == wr_ptr) begin
                            // No more events at all
                            valid <= 1'b1;
                        end else begin
                            state <= S_WAIT;
                        end
                    end
                end
                
                S_WAIT: begin
                    // Wait one cycle for BRAM read to complete
                    state <= S_READ;
                end
                
                S_READ: begin
                    if (mem_dout[15:10] == query_t) begin
                        spike_bus_out[mem_dout[9:0]] <= 1'b1;
                        rd_ptr <= rd_ptr + 1;
                        cur_rd <= cur_rd + 1;
                        if (rd_ptr + 1 == wr_ptr) begin
                            // Reached end of recorded events
                            valid <= 1'b1;
                            state <= S_IDLE;
                        end else begin
                            // fetch next
                            state <= S_WAIT;
                        end
                    end else begin
                        // Found an event for a future timestep
                        valid <= 1'b1;
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
