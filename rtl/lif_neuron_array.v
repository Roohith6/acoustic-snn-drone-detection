// =============================================================================
// lif_neuron_array.v
// -----------------------------------------------------------------------------
// Genuine LIF (IF) hidden layer.
// Wraps spike_gated_mac and v_mem.
// After MAC finishes, iterates through neurons, updates V, and fires spikes.
// =============================================================================
module lif_neuron_array #(
    parameter N_NEURONS = 640,
    parameter N_POST    = 32,
    parameter W1_MEM_FILE = "",
    parameter B1_MEM_FILE = "",
    parameter V_THRESH_FILE = ""
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                    start,
    input  wire [N_NEURONS-1:0]    spike_bus,
    input  wire                    timestep_valid,
    output reg                     busy,
    
    output reg  [N_POST-1:0]       out_spike_bus,
    output reg                     out_valid
);

    wire mac_busy;
    reg  [$clog2(N_POST)-1:0] rd_addr;
    wire signed [31:0] raw_acc_data;

    spike_gated_mac #(
        .N_NEURONS(N_NEURONS), .N_POST(N_POST), .WEIGHT_WIDTH(16), .ACC_WIDTH(32),
        .WEIGHT_MEM_FILE(W1_MEM_FILE)
    ) u_mac (
        .clk(clk), .rst_n(rst_n),
        .start(timestep_valid), .spike_bus(spike_bus), .timestep_valid(timestep_valid), .busy(mac_busy),
        .acc_rd_addr(rd_addr), .acc_rd_data(raw_acc_data)
    );

    wire signed [31:0] v_mem_rd_data;
    reg  v_mem_wr_en;
    reg  [$clog2(N_POST)-1:0] v_mem_wr_addr;
    reg  signed [31:0] v_mem_wr_data;

    v_mem #(
        .N_NEURONS(N_POST), .WIDTH(32)
    ) u_vmem (
        .clk(clk), .rst_n(rst_n), .start(start),
        .wr_en(v_mem_wr_en), .wr_addr(v_mem_wr_addr), .wr_data(v_mem_wr_data),
        .rd_addr(rd_addr), .rd_data(v_mem_rd_data)
    );

    reg signed [31:0] v_thresh;
    initial begin
        if (V_THRESH_FILE != "") begin
            $readmemh(V_THRESH_FILE, thresh_mem);
        end
        if (B1_MEM_FILE != "") begin
            $readmemh(B1_MEM_FILE, bias_mem);
        end
    end
    reg signed [31:0] thresh_mem [0:0];
    reg signed [31:0] bias_mem [0:N_POST-1];

    always @(*) begin
        v_thresh = thresh_mem[0];
    end

    localparam S_IDLE      = 3'd0;
    localparam S_START     = 3'd1;  // one cycle after timestep_valid for MAC to latch
    localparam S_BUSY      = 3'd2;  // wait for MAC to finish
    localparam S_READ_MEM  = 3'd3;
    localparam S_EVAL_WRITE= 3'd4;

    reg [2:0] state;
    reg [$clog2(N_POST):0] n_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            out_valid <= 1'b0;
            out_spike_bus <= 0;
            rd_addr <= 0;
            v_mem_wr_en <= 1'b0;
            v_mem_wr_addr <= 0;
            v_mem_wr_data <= 0;
            state <= S_IDLE;
            n_cnt <= 0;
        end else begin
            out_valid <= 1'b0;
            v_mem_wr_en <= 1'b0;

            if (start) begin
                out_spike_bus <= 0;
                state <= S_IDLE;
                busy <= 1'b0;
            end

            case (state)
                S_IDLE: begin
                    if (timestep_valid) begin
                        busy <= 1'b1;
                        out_spike_bus <= 0;
                        state <= S_START;
                    end
                end

                S_START: begin
                    // MAC has latched spikes, advance to wait for completion
                    state <= S_BUSY;
                end

                S_BUSY: begin
                    if (!mac_busy) begin
                        n_cnt   <= 0;
                        rd_addr <= 0;
                        state   <= S_READ_MEM;
                    end
                end

                S_READ_MEM: begin
                    // Read addresses are set. Wait 1 cycle for data.
                    state <= S_EVAL_WRITE;
                end

                S_EVAL_WRITE: begin
                    // Data is valid.
                    // V_new = V + raw_acc + bias
                    // If V_new >= V_thresh -> spike, V_new = V_new - V_thresh
                    if (v_mem_rd_data + raw_acc_data + bias_mem[n_cnt] >= v_thresh) begin
                        out_spike_bus[n_cnt] <= 1'b1;
                        v_mem_wr_data <= v_mem_rd_data + raw_acc_data + bias_mem[n_cnt] - v_thresh;
                    end else begin
                        out_spike_bus[n_cnt] <= 1'b0;
                        // Avoid negative potential accumulation for stability (ReLU equivalent)
                        if (v_mem_rd_data + raw_acc_data + bias_mem[n_cnt] < 0)
                            v_mem_wr_data <= 0;
                        else
                            v_mem_wr_data <= v_mem_rd_data + raw_acc_data + bias_mem[n_cnt];
                    end
                    v_mem_wr_addr <= n_cnt;
                    v_mem_wr_en <= 1'b1;

                    if (n_cnt == N_POST - 1) begin
                        out_valid <= 1'b1;
                        busy <= 1'b0;
                        state <= S_IDLE;
                    end else begin
                        n_cnt <= n_cnt + 1;
                        rd_addr <= n_cnt + 1;
                        state <= S_READ_MEM;
                    end
                end
            endcase
        end
    end
endmodule
