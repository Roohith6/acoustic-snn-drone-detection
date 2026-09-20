// =============================================================================
// snn_top.v
// -----------------------------------------------------------------------------
// Full inference pipeline, single interface: assert `start`, wait for `done`,
// read `winner`. Internally sequences four phases that were previously
// driven by hand in the testbenches:
//
//   PHASE 1 (ENCODE):  run rate_encoder for one full pass (32000 spike_valid
//                       pulses), buffering every bit into spike_storage.
//                       NOTE: rate_encoder has no pause/backpressure input,
//                       so it always runs a full pass at its own pace; this
//                       phase decouples it from the (slower) accumulate
//                       phase rather than trying to add stall logic to an
//                       already-verified module. Simplest-correct-first,
//                       flagged as a real design choice, not an oversight.
//   PHASE 2 (HIDDEN):   feed spike_storage into hidden_neuron, one timestep
//                       at a time (50x), exactly as tb_hidden_neuron.v did
//                       by hand.
//   PHASE 3 (OUTPUT):   stream all 128 (h, a1) pairs into output_layer,
//                       exactly as tb_output_layer.v did by hand. The wait
//                       between changing rd_addr and a1_out being valid is
//                       a FIXED, KNOWN 4-cycle pipeline latency (1 memory
//                       read + 1 combine_scale + 2 sigmoid_lut stages) --
//                       used here as a hardcoded wait count rather than a
//                       real valid/ready handshake. Documented because this
//                       is brittle to changes in those submodules' internal
//                       pipeline depth -- a real handshake signal would be
//                       the correct fix if those modules change.
//   PHASE 4 (FINALIZE): pulse output_layer's finalize, wait for its done.
// =============================================================================
module snn_top #(
    parameter N_NEURONS = 640,
    parameter N_POST    = 128,
    parameter N_STEPS   = 50,
    parameter THRESH_MEM_FILE   = "",
    parameter W1_MEM_FILE       = "",
    parameter B1_MEM_FILE       = "",
    parameter SIGMOID_MEM_FILE  = "",
    parameter W2_MEM_FILE       = "",
    parameter B2_MEM_FILE       = ""
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output reg  winner,           // 0 = ambient, 1 = drone
    output reg  signed [63:0] z2_total0,
    output reg  signed [63:0] z2_total1
);

    localparam HIDDEN_READ_LATENCY = 4;

    // ---- rate encoder ----
    reg  enc_start;
    wire enc_spike_valid, enc_spike_bit, enc_done;
    wire [$clog2(N_STEPS)-1:0]   enc_out_t;
    wire [$clog2(N_NEURONS)-1:0] enc_out_n;

    rate_encoder #(
        .N_NEURONS(N_NEURONS), .N_STEPS(N_STEPS), .WIDTH(16), .SEED(16'hACE1),
        .THRESH_MEM_FILE(THRESH_MEM_FILE)
    ) u_encoder (
        .clk(clk), .rst_n(rst_n), .start(enc_start),
        .spike_valid(enc_spike_valid), .spike_bit(enc_spike_bit),
        .out_t(enc_out_t), .out_n(enc_out_n), .done(enc_done)
    );

    // ---- spike storage: buffers one full pass (50 x 640 bits) ----
    reg [N_NEURONS-1:0] spike_storage [0:N_STEPS-1];
    integer si;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (si = 0; si < N_STEPS; si = si + 1)
                spike_storage[si] <= 0;
        end else if (enc_spike_valid) begin
            spike_storage[enc_out_t][enc_out_n] <= enc_spike_bit;
        end
    end

    // ---- hidden layer ----
    reg  hid_start;
    reg  [N_NEURONS-1:0] hid_spike_bus;
    reg  hid_timestep_valid;
    wire hid_busy;
    reg  [$clog2(N_POST)-1:0] hid_rd_addr;
    wire [15:0] hid_a1_out;

    hidden_neuron #(
        .N_NEURONS(N_NEURONS), .N_POST(N_POST),
        .W1_MEM_FILE(W1_MEM_FILE), .B1_MEM_FILE(B1_MEM_FILE), .SIGMOID_MEM_FILE(SIGMOID_MEM_FILE)
    ) u_hidden (
        .clk(clk), .rst_n(rst_n),
        .start(hid_start), .spike_bus(hid_spike_bus), .timestep_valid(hid_timestep_valid), .busy(hid_busy),
        .rd_addr(hid_rd_addr), .a1_out(hid_a1_out)
    );

    // ---- output layer ----
    reg  out_start;
    reg  [$clog2(N_POST)-1:0] out_h_index;
    reg  [15:0] out_a1_value;
    reg  out_h_valid;
    wire out_busy;
    reg  out_finalize;
    wire out_done;
    wire out_winner;
    wire signed [63:0] out_z2_total0, out_z2_total1;

    output_layer #(
        .N_POST(N_POST), .N_OUT(2),
        .W2_MEM_FILE(W2_MEM_FILE), .B2_MEM_FILE(B2_MEM_FILE)
    ) u_output (
        .clk(clk), .rst_n(rst_n),
        .start(out_start), .h_index(out_h_index), .a1_value(out_a1_value), .h_valid(out_h_valid), .busy(out_busy),
        .finalize(out_finalize), .done(out_done), .winner(out_winner),
        .z2_total0(out_z2_total0), .z2_total1(out_z2_total1)
    );

    // ---- master FSM ----
    localparam S_IDLE            = 5'd0;
    localparam S_ENCODE_START    = 5'd1;
    localparam S_ENCODE_WAIT     = 5'd2;
    localparam S_HIDDEN_START    = 5'd3;
    localparam S_HIDDEN_SET_BUS  = 5'd4;
    localparam S_HIDDEN_PULSE    = 5'd5;
    localparam S_HIDDEN_WAIT_BUSY_HIGH = 5'd6;
    localparam S_HIDDEN_WAIT     = 5'd7;
    localparam S_OUTPUT_START    = 5'd8;
    localparam S_OUTPUT_SET_ADDR = 5'd9;
    localparam S_OUTPUT_WAIT_LAT = 5'd10;
    localparam S_OUTPUT_PULSE    = 5'd11;
    localparam S_OUTPUT_WAIT_BUSY_HIGH = 5'd12;
    localparam S_OUTPUT_WAIT     = 5'd13;
    localparam S_FINALIZE        = 5'd14;
    localparam S_WAIT_FINAL_DONE = 5'd15;
    localparam S_DONE            = 5'd16;

    reg [4:0] state;
    reg [$clog2(N_STEPS)-1:0]  t_cnt;
    reg [$clog2(N_POST)-1:0]   h_cnt;
    reg [3:0] lat_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done  <= 1'b0;
            enc_start <= 1'b0;
            hid_start <= 1'b0; hid_timestep_valid <= 1'b0;
            out_start <= 1'b0; out_h_valid <= 1'b0; out_finalize <= 1'b0;
            t_cnt <= 0; h_cnt <= 0; lat_cnt <= 0;
        end else begin
            done <= 1'b0;
            enc_start <= 1'b0;
            hid_timestep_valid <= 1'b0;
            out_h_valid <= 1'b0;
            out_finalize <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        enc_start <= 1'b1;
                        state <= S_ENCODE_WAIT;
                    end
                end

                S_ENCODE_WAIT: begin
                    if (enc_done) begin
                        hid_start <= 1'b1;
                        t_cnt <= 0;
                        state <= S_HIDDEN_SET_BUS;
                    end
                end

                S_HIDDEN_SET_BUS: begin
                    hid_start <= 1'b0;
                    hid_spike_bus <= spike_storage[t_cnt];
                    state <= S_HIDDEN_PULSE;
                end

                S_HIDDEN_PULSE: begin
                    hid_timestep_valid <= 1'b1;
                    state <= S_HIDDEN_WAIT_BUSY_HIGH;
                end

                S_HIDDEN_WAIT_BUSY_HIGH: begin
                    // guard against checking !hid_busy before spike_gated_mac
                    // has actually had a chance to raise it -- caught by
                    // simulation (totals came out too LARGE, the signature
                    // of a premature re-trigger/over-accumulation, not a
                    // missed one). See PROJECT_REPORT.md.
                    if (hid_busy)
                        state <= S_HIDDEN_WAIT;
                end

                S_HIDDEN_WAIT: begin
                    if (!hid_busy) begin
                        if (t_cnt == N_STEPS-1) begin
                            out_start <= 1'b1;
                            h_cnt <= 0;
                            state <= S_OUTPUT_SET_ADDR;
                        end else begin
                            t_cnt <= t_cnt + 1'b1;
                            state <= S_HIDDEN_SET_BUS;
                        end
                    end
                end

                S_OUTPUT_SET_ADDR: begin
                    out_start <= 1'b0;
                    hid_rd_addr <= h_cnt;
                    lat_cnt <= 0;
                    state <= S_OUTPUT_WAIT_LAT;
                end

                S_OUTPUT_WAIT_LAT: begin
                    if (lat_cnt == HIDDEN_READ_LATENCY) begin
                        out_a1_value <= hid_a1_out;
                        out_h_index  <= h_cnt;
                        state <= S_OUTPUT_PULSE;
                    end else begin
                        lat_cnt <= lat_cnt + 1'b1;
                    end
                end

                S_OUTPUT_PULSE: begin
                    out_h_valid <= 1'b1;
                    state <= S_OUTPUT_WAIT_BUSY_HIGH;
                end

                S_OUTPUT_WAIT_BUSY_HIGH: begin
                    if (out_busy)
                        state <= S_OUTPUT_WAIT;
                end

                S_OUTPUT_WAIT: begin
                    if (!out_busy) begin
                        if (h_cnt == N_POST-1) begin
                            state <= S_FINALIZE;
                        end else begin
                            h_cnt <= h_cnt + 1'b1;
                            state <= S_OUTPUT_SET_ADDR;
                        end
                    end
                end

                S_FINALIZE: begin
                    out_finalize <= 1'b1;
                    state <= S_WAIT_FINAL_DONE;
                end

                S_WAIT_FINAL_DONE: begin
                    if (out_done) begin
                        winner    <= out_winner;
                        z2_total0 <= out_z2_total0;
                        z2_total1 <= out_z2_total1;
                        state <= S_DONE;
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
