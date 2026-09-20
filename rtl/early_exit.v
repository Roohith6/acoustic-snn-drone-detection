// =============================================================================
// early_exit.v
// -----------------------------------------------------------------------------
// Adaptive Temporal Early-Exit mechanism.
// Monitors output LIF spike counts and exits if the difference exceeds a threshold.
// =============================================================================
module early_exit #(
    parameter CONF_THRESH = 10,
    parameter MIN_STEPS   = 5
) (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] spike_cnt_drone,
    input  wire [7:0] spike_cnt_ambient,
    input  wire [5:0] t_current,
    input  wire check_now,     // pulse from FSM to evaluate condition
    output reg  exit_now       // registered output for clean FSM branching
);

    wire [7:0] diff_0 = spike_cnt_drone - spike_cnt_ambient;
    wire [7:0] diff_1 = spike_cnt_ambient - spike_cnt_drone;
    wire [7:0] abs_diff = (spike_cnt_drone > spike_cnt_ambient) ? diff_0 : diff_1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exit_now <= 1'b0;
        end else begin
            if (check_now) begin
                if ((t_current >= MIN_STEPS) && (abs_diff >= CONF_THRESH)) begin
                    exit_now <= 1'b1;
                end else begin
                    exit_now <= 1'b0;
                end
            end else begin
                exit_now <= 1'b0;
            end
        end
    end

endmodule
