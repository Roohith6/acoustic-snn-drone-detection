// =============================================================================
// hidden_neuron.v
// -----------------------------------------------------------------------------
// Wires together the verified pieces into the actual hidden-layer
// computation: spike_gated_mac (raw_acc) + b1 memory (both read by the same
// rd_addr, same 1-cycle latency, so they land together) -> combine_scale
// (z1_q) -> sigmoid_lut (a1 = sigmoid(z1_q)). Per-hidden-neuron output is
// read out one at a time via rd_addr/a1_out.
// =============================================================================
module hidden_neuron #(
    parameter N_NEURONS = 640,
    parameter N_POST    = 128,
    parameter W1_MEM_FILE = "",
    parameter B1_MEM_FILE = "",
    parameter SIGMOID_MEM_FILE = ""
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                    start,
    input  wire [N_NEURONS-1:0]    spike_bus,
    input  wire                    timestep_valid,
    output wire                    busy,

    input  wire [$clog2(N_POST)-1:0] rd_addr,   // which hidden neuron to read
    output wire [15:0]               a1_out     // sigmoid(z1) for that neuron, Q0.16
);

    wire signed [31:0] raw_acc_data;

    spike_gated_mac #(
        .N_NEURONS(N_NEURONS), .N_POST(N_POST), .WEIGHT_WIDTH(16), .ACC_WIDTH(32),
        .WEIGHT_MEM_FILE(W1_MEM_FILE)
    ) u_mac (
        .clk(clk), .rst_n(rst_n),
        .start(start), .spike_bus(spike_bus), .timestep_valid(timestep_valid), .busy(busy),
        .acc_rd_addr(rd_addr), .acc_rd_data(raw_acc_data)
    );

    wire signed [15:0] b1_data;
    weight_mem #(
        .NUM_PRE(1), .NUM_POST(N_POST), .WIDTH(16), .MEM_FILE(B1_MEM_FILE)
    ) u_b1 (
        .clk(clk), .addr(rd_addr), .data_out(b1_data)
    );

    wire signed [15:0] z1_q;
    combine_scale u_combine (
        .clk(clk),
        .raw_acc(raw_acc_data),
        .b1_int(b1_data),
        .z1_q(z1_q)
    );

    sigmoid_lut #(
        .FRAC_BITS(7), .TABLE_SIZE(2048), .MEM_FILE(SIGMOID_MEM_FILE)
    ) u_sigmoid (
        .clk(clk), .z1_q(z1_q), .sigmoid_out(a1_out)
    );

endmodule
