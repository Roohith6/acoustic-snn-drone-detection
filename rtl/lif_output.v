// =============================================================================
// lif_output.v
// -----------------------------------------------------------------------------
// Two LIF output neurons.
// Instantiates spike_gated_mac (N_NEURONS=128, N_POST=2).
// Tracks spike counts over ALL timesteps (start resets across a full inference,
// not per-timestep -- spike counts accumulate for the whole inference).
// =============================================================================
module lif_output #(
    parameter N_PRE = 32,
    parameter N_OUT = 2,
    parameter W2_MEM_FILE = "",
    parameter B2_MEM_FILE = "",
    parameter V_THRESH_FILE = ""
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                    start,          // pulse at start of inference -- resets
    input  wire [N_PRE-1:0]        spike_bus,
    input  wire                    timestep_valid, // pulse once per timestep
    output reg                     busy,
    
    // Outputs available after each timestep evaluation
    output wire [7:0]              spike_cnt_ambient,
    output wire [7:0]              spike_cnt_drone,
    output reg                     done,
    output reg                     winner
);

    // MAC: start=start (inference reset, clears accumulators once per inference start)
    //       timestep_valid triggers a new per-timestep accumulation
    wire mac_busy;
    reg  [$clog2(N_OUT)-1:0] rd_addr;
    wire signed [31:0] raw_acc_data;

    // NOTE: MAC's 'start' clears its accumulator. We pass the inference 'start'
    // so it is cleared once at the beginning. Then each timestep_valid triggers
    // the MAC to accumulate the hidden spikes for that timestep only.
    spike_gated_mac #(
        .N_NEURONS(N_PRE), .N_POST(N_OUT), .WEIGHT_WIDTH(16), .ACC_WIDTH(32),
        .WEIGHT_MEM_FILE(W2_MEM_FILE)
    ) u_mac (
        .clk(clk), .rst_n(rst_n),
        .start(timestep_valid),      // clear acc every timestep
        .spike_bus(spike_bus), .timestep_valid(timestep_valid), .busy(mac_busy),
        .acc_rd_addr(rd_addr), .acc_rd_data(raw_acc_data)
    );

    reg signed [31:0] v_thresh;
    initial begin
        if (V_THRESH_FILE != "") begin
            $readmemh(V_THRESH_FILE, thresh_mem);
        end
        if (B2_MEM_FILE != "") begin
            $readmemh(B2_MEM_FILE, bias_mem);
        end
    end
    reg signed [31:0] thresh_mem [0:0];
    reg signed [31:0] bias_mem [0:N_OUT-1];
    always @(*) begin
        v_thresh = thresh_mem[0];
    end

    reg signed [31:0] v_0;
    reg signed [31:0] v_1;
    reg [7:0] cnt_0;
    reg [7:0] cnt_1;
    
    assign spike_cnt_ambient = cnt_0;
    assign spike_cnt_drone = cnt_1;

    localparam S_IDLE   = 3'd0;
    localparam S_START  = 3'd1;  // wait one cycle after timestep_valid for MAC to latch
    localparam S_BUSY   = 3'd2;  // wait for MAC to finish
    localparam S_READ_0 = 3'd3;
    localparam S_EVAL_0 = 3'd4;
    localparam S_READ_1 = 3'd5;
    localparam S_EVAL_1 = 3'd6;

    reg [2:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            done    <= 1'b0;
            winner  <= 1'b0;
            v_0     <= 0;
            v_1     <= 0;
            cnt_0   <= 0;
            cnt_1   <= 0;
            state   <= S_IDLE;
            rd_addr <= 0;
        end else begin
            done <= 1'b0;

            // Inference-level reset
            if (start) begin
                v_0   <= 0;
                v_1   <= 0;
                cnt_0 <= 0;
                cnt_1 <= 0;
                state <= S_IDLE;
                busy  <= 1'b0;
            end

            case (state)
                S_IDLE: begin
                    if (timestep_valid) begin
                        busy  <= 1'b1;
                        state <= S_START;
                    end
                end

                S_START: begin
                    // MAC has now latched spike_bus. Wait for it to finish.
                    state <= S_BUSY;
                end

                S_BUSY: begin
                    if (!mac_busy) begin
                        // MAC done. Set up read for neuron 0.
                        rd_addr <= 0;
                        state   <= S_READ_0;
                    end
                end

                S_READ_0: begin
                    // rd_addr=0 is registered, data arrives next cycle
                    rd_addr <= 1;        // prep read for neuron 1
                    state   <= S_EVAL_0;
                end

                S_EVAL_0: begin
                    // raw_acc_data = acc[0] (from rd_addr=0 set in S_READ_0)
                    if (v_0 + raw_acc_data + bias_mem[0] >= v_thresh) begin
                        v_0   <= v_0 + raw_acc_data + bias_mem[0] - v_thresh;
                        cnt_0 <= cnt_0 + 1;
                    end else begin
                        if (v_0 + raw_acc_data + bias_mem[0] < 0) v_0 <= 0;
                        else v_0 <= v_0 + raw_acc_data + bias_mem[0];
                    end
                    state <= S_READ_1;
                end

                S_READ_1: begin
                    // rd_addr=1 was set in S_READ_0; data now valid
                    // (raw_acc_data = acc[1])
                    state <= S_EVAL_1;
                end

                S_EVAL_1: begin
                    // raw_acc_data = acc[1] (from rd_addr=1 set in S_READ_0)
                    if (v_1 + raw_acc_data + bias_mem[1] >= v_thresh) begin
                        v_1   <= v_1 + raw_acc_data + bias_mem[1] - v_thresh;
                        cnt_1 <= cnt_1 + 1;
                    end else begin
                        if (v_1 + raw_acc_data + bias_mem[1] < 0) v_1 <= 0;
                        else v_1 <= v_1 + raw_acc_data + bias_mem[1];
                    end
                    
                    busy   <= 1'b0;
                    winner <= (cnt_1 > cnt_0) ? 1'b1 : 1'b0;
                    done   <= 1'b1;
                    state  <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
