// =============================================================================
// spike_gated_mac.v
// -----------------------------------------------------------------------------
// Accumulates z1 (pre-bias, pre-average) for all N_POST hidden neurons from
// one timestep's spike vector: for every input neuron n with spike_bus[n]==1,
// adds W1's row n (N_POST wide) into the running accumulators. Non-firing
// neurons are skipped entirely -- no memory read, no add -- which is the
// actual resource-saving mechanism the sparsity numbers in the original
// project doc (drone ~14%, ambient ~5% active) are meant to exploit.
//
// Call once per timestep with a fresh spike_bus + timestep_valid pulse;
// wait for !busy before issuing the next timestep. After all N_STEPS calls,
// read out the 128 accumulators via acc_rd_addr/acc_rd_data (combinational-
// read-friendly: registered like the other memories in this design).
//
// "Simplest correct" first pass, per project priority order: sequential,
// one weight read per cycle (steady state), no attempt yet to parallelize
// or widen the memory word -- documented as a known optimization target,
// not a currently-hidden limitation.
// =============================================================================
module spike_gated_mac #(
    parameter N_NEURONS  = 640,
    parameter N_POST     = 128,
    parameter WEIGHT_WIDTH = 16,
    parameter ACC_WIDTH  = 32,
    parameter WEIGHT_MEM_FILE = ""
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    start,          // clear all accumulators to 0
    input  wire [N_NEURONS-1:0]    spike_bus,       // this timestep's 640 spike bits
    input  wire                    timestep_valid,  // pulse: consume spike_bus now
    output reg                     busy,            // high while processing a timestep

    input  wire [$clog2(N_POST)-1:0] acc_rd_addr,
    output reg  signed [ACC_WIDTH-1:0] acc_rd_data
);

    localparam NBITS_N = $clog2(N_NEURONS);
    localparam NBITS_H = $clog2(N_POST);

    reg signed [ACC_WIDTH-1:0] acc [0:N_POST-1];

    reg [N_NEURONS-1:0] spike_latched;
    reg [NBITS_N-1:0]   n_cnt;
    reg [NBITS_H-1:0]   h_cnt;
    reg                 in_row;      // currently walking an active row's N_POST weights

    // pipeline: 1-cycle registered weight_mem read, correlate with which (n,h) it was for
    reg                 issuing;
    reg [NBITS_N-1:0]   n_issue;
    reg [NBITS_H-1:0]   h_issue;
    reg                 issuing_d;
    reg [NBITS_H-1:0]   h_capture;

    wire [$clog2(N_NEURONS*N_POST)-1:0] w_addr = n_issue * N_POST + h_issue;
    wire signed [WEIGHT_WIDTH-1:0] w_data;

    weight_mem #(
        .NUM_PRE(N_NEURONS), .NUM_POST(N_POST), .WIDTH(WEIGHT_WIDTH),
        .MEM_FILE(WEIGHT_MEM_FILE)
    ) u_w1 (
        .clk(clk),
        .addr(w_addr),
        .data_out(w_data)
    );

    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy          <= 1'b0;
            n_cnt         <= 0;
            h_cnt         <= 0;
            in_row        <= 1'b0;
            issuing       <= 1'b0;
            issuing_d     <= 1'b0;
            n_issue       <= 0;
            h_issue       <= 0;
            h_capture     <= 0;
            spike_latched <= 0;
            for (k = 0; k < N_POST; k = k + 1)
                acc[k] <= 0;
        end else begin

            if (start) begin
                for (k = 0; k < N_POST; k = k + 1)
                    acc[k] <= 0;
            end

            if (timestep_valid && !busy) begin
                spike_latched <= spike_bus;
                n_cnt   <= 0;
                h_cnt   <= 0;
                in_row  <= 1'b0;
                busy    <= 1'b1;
                issuing <= 1'b0;
            end else if (busy) begin
                if (!in_row) begin
                    // scanning for the next active input neuron
                    if (n_cnt == N_NEURONS) begin
                        // finished all 640 -- but let the pipeline drain first
                        if (!issuing && !issuing_d) begin
                            busy <= 1'b0;
                        end
                        issuing <= 1'b0;
                    end else if (spike_latched[n_cnt]) begin
                        in_row  <= 1'b1;
                        h_cnt   <= 0;
                        // NOTE: do not issue here -- the in_row branch issues
                        // h=0 naturally next cycle. Issuing here too caused
                        // h=0 to be read/added twice per active row (caught
                        // by simulation: h=0 came out exactly 2x every other
                        // index on both real samples -- see PROJECT_REPORT.md).
                        issuing <= 1'b0;
                    end else begin
                        n_cnt   <= n_cnt + 1'b1;
                        issuing <= 1'b0;
                    end
                end else begin
                    // walking an active row's N_POST weights
                    issuing <= 1'b1;
                    n_issue <= n_cnt;
                    h_issue <= h_cnt;
                    if (h_cnt == N_POST-1) begin
                        in_row <= 1'b0;
                        n_cnt  <= n_cnt + 1'b1;
                    end else begin
                        h_cnt <= h_cnt + 1'b1;
                    end
                end
            end

            // pipeline capture: 1 cycle after issuing, w_data is valid for (n_issue,h_issue)
            issuing_d <= issuing;
            h_capture <= h_issue;
            if (issuing_d) begin
                acc[h_capture] <= acc[h_capture] + w_data;
            end
        end
    end

    always @(posedge clk) begin
        acc_rd_data <= acc[acc_rd_addr];
    end

endmodule
