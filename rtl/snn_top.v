// =============================================================================
// snn_top.v
// -----------------------------------------------------------------------------
// LIF SNN with Sparse AER and Adaptive Temporal Early-Exit.
// =============================================================================
module snn_top #(
    parameter N_NEURONS = 640,
    parameter N_STEPS   = 50,
    parameter N_POST    = 32,
    parameter W1_MEM_FILE = "",
    parameter W2_MEM_FILE = "",
    parameter B1_MEM_FILE = "",
    parameter B2_MEM_FILE = "",
    parameter V_THRESH_HIDDEN = "",
    parameter V_THRESH_OUTPUT = "",
    parameter THRESH_MEM_FILE_0 = "",
    parameter THRESH_MEM_FILE_1 = "",
    parameter THRESH_MEM_FILE_2 = "",
    parameter THRESH_MEM_FILE_3 = "",
    parameter CONF_THRESH = 10,
    parameter MIN_STEPS = 5
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [1:0] sample_sel,
    output reg  done,
    output reg  winner,           // 0 = ambient, 1 = drone
    output reg  [7:0] spike_cnt_ambient,
    output reg  [7:0] spike_cnt_drone,
    output reg  [5:0] t_exit      // timestep when it exited
);

    // ---- rate encoder ----
    reg  enc_start;
    wire enc_spike_valid, enc_spike_bit, enc_done;
    wire [$clog2(N_STEPS)-1:0]   enc_out_t;
    wire [$clog2(N_NEURONS)-1:0] enc_out_n;

    rate_encoder #(
        .N_NEURONS(N_NEURONS), .N_STEPS(N_STEPS), .WIDTH(16), .SEED(16'hACE1),
        .THRESH_MEM_FILE_0(THRESH_MEM_FILE_0), .THRESH_MEM_FILE_1(THRESH_MEM_FILE_1),
        .THRESH_MEM_FILE_2(THRESH_MEM_FILE_2), .THRESH_MEM_FILE_3(THRESH_MEM_FILE_3)
    ) u_encoder (
        .clk(clk), .rst_n(rst_n), .start(enc_start), .sample_sel(sample_sel),
        .spike_valid(enc_spike_valid), .spike_bit(enc_spike_bit),
        .out_t(enc_out_t), .out_n(enc_out_n), .done(enc_done)
    );

    // ---- AER buffer ----
    reg aer_start;
    reg aer_fetch;
    wire [N_NEURONS-1:0] aer_bus_out;
    wire aer_valid;
    reg [5:0] t_cnt;

    aer_buffer #(
        .MAX_EVENTS(32768), .N_NEURONS(N_NEURONS), .N_STEPS(N_STEPS)
    ) u_aer (
        .clk(clk), .rst_n(rst_n), .start(aer_start),
        .wr_en(enc_spike_valid && enc_spike_bit), .wr_t(enc_out_t), .wr_n(enc_out_n),
        .fetch(aer_fetch), .query_t(t_cnt), .spike_bus_out(aer_bus_out), .valid(aer_valid)
    );

    // ---- hidden LIF array ----
    reg hid_start;
    reg hid_ts_valid;
    wire hid_busy;
    wire [N_POST-1:0] hid_spike_out;
    wire hid_out_valid;

    lif_neuron_array #(
        .N_NEURONS(N_NEURONS), .N_POST(N_POST), 
        .W1_MEM_FILE(W1_MEM_FILE), .B1_MEM_FILE(B1_MEM_FILE), .V_THRESH_FILE(V_THRESH_HIDDEN)
    ) u_hidden (
        .clk(clk), .rst_n(rst_n), .start(hid_start), 
        .spike_bus(aer_bus_out), .timestep_valid(hid_ts_valid), .busy(hid_busy),
        .out_spike_bus(hid_spike_out), .out_valid(hid_out_valid)
    );

    // ---- output LIF ----
    reg out_start;
    reg out_ts_valid;
    wire out_busy;
    wire [7:0] out_cnt_ambient;
    wire [7:0] out_cnt_drone;
    wire out_done;
    wire out_winner;

    lif_output #(
        .N_PRE(N_POST), .N_OUT(2),
        .W2_MEM_FILE(W2_MEM_FILE), .B2_MEM_FILE(B2_MEM_FILE), .V_THRESH_FILE(V_THRESH_OUTPUT)
    ) u_output (
        .clk(clk), .rst_n(rst_n), .start(out_start),
        .spike_bus(hid_spike_out), .timestep_valid(out_ts_valid), .busy(out_busy),
        .spike_cnt_ambient(out_cnt_ambient), .spike_cnt_drone(out_cnt_drone),
        .done(out_done), .winner(out_winner)
    );

    // ---- early exit ----
    reg early_check_now;
    wire early_exit_now;
    early_exit #(
        .CONF_THRESH(CONF_THRESH), .MIN_STEPS(MIN_STEPS)
    ) u_early (
        .clk(clk), .rst_n(rst_n),
        .spike_cnt_drone(out_cnt_drone), .spike_cnt_ambient(out_cnt_ambient),
        .t_current(t_cnt), .check_now(early_check_now), .exit_now(early_exit_now)
    );

    // ---- master FSM ----
    localparam S_IDLE            = 4'd0;
    localparam S_ENCODE_WAIT     = 4'd1;
    localparam S_AER_FETCH       = 4'd2;
    localparam S_AER_WAIT        = 4'd3;
    localparam S_HIDDEN_PULSE    = 4'd4;
    localparam S_HIDDEN_WAIT     = 4'd5;
    localparam S_OUTPUT_PULSE    = 4'd6;
    localparam S_OUTPUT_WAIT     = 4'd7;
    localparam S_EARLY_CHECK     = 4'd8;
    localparam S_EARLY_WAIT      = 4'd9;
    localparam S_DONE            = 4'd10;

    reg [3:0] state;
    reg hid_busy_d;
    reg out_busy_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            enc_start <= 1'b0;
            aer_start <= 1'b0; aer_fetch <= 1'b0;
            hid_start <= 1'b0; hid_ts_valid <= 1'b0;
            out_start <= 1'b0; out_ts_valid <= 1'b0;
            early_check_now <= 1'b0;
            t_cnt <= 0;
            hid_busy_d <= 1'b0;
            out_busy_d <= 1'b0;
        end else begin
            done <= 1'b0;
            enc_start <= 1'b0; aer_fetch <= 1'b0; aer_start <= 1'b0;
            hid_start <= 1'b0; hid_ts_valid <= 1'b0;
            out_start <= 1'b0; out_ts_valid <= 1'b0;
            early_check_now <= 1'b0;
            
            hid_busy_d <= hid_busy;
            out_busy_d <= out_busy;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        enc_start <= 1'b1;
                        aer_start <= 1'b1;
                        hid_start <= 1'b1;
                        out_start <= 1'b1;
                        state <= S_ENCODE_WAIT;
                    end
                end

                S_ENCODE_WAIT: begin
                    if (enc_done) begin
                        t_cnt <= 0;
                        aer_fetch <= 1'b1;
                        state <= S_AER_FETCH;
                    end
                end

                S_AER_FETCH: begin
                    state <= S_AER_WAIT;
                end

                S_AER_WAIT: begin
                    if (aer_valid) begin
                        hid_ts_valid <= 1'b1;
                        state <= S_HIDDEN_WAIT;
                    end
                end

                S_HIDDEN_WAIT: begin
                    // Wait for hid_busy to fall or hid_out_valid
                    if (hid_out_valid) begin
                        out_ts_valid <= 1'b1;
                        state <= S_OUTPUT_WAIT;
                    end
                end

                S_OUTPUT_WAIT: begin
                    if (out_done) begin
                        early_check_now <= 1'b1;
                        state <= S_EARLY_WAIT;
                    end
                end

                S_EARLY_WAIT: begin
                    if (early_exit_now || t_cnt == N_STEPS - 1) begin
                        winner <= out_winner;
                        spike_cnt_ambient <= out_cnt_ambient;
                        spike_cnt_drone <= out_cnt_drone;
                        t_exit <= t_cnt;
                        state <= S_DONE;
                    end else begin
                        t_cnt <= t_cnt + 1;
                        aer_fetch <= 1'b1;
                        state <= S_AER_FETCH;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
