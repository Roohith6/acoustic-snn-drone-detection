// =============================================================================
// rate_encoder.v
// -----------------------------------------------------------------------------
// Reproduces deterministic_rate_encode() from audio_to_spike.py bit-exactly:
//   for t in 0..N_STEPS-1:
//     for n in 0..N_NEURONS-1:
//       rand_val = lfsr.next()
//       spike[t,n] = 1 if rand_val < threshold_int[n] else 0
//
// One LFSR draw + one threshold comparison per clock cycle (steady state).
// LFSR and threshold_mem are both 1-cycle registered reads, so this is a
// simple 2-stage pipeline: cycle k issues the (t,n) request, cycle k+1
// captures the comparison result for that same (t,n).
// =============================================================================
module rate_encoder #(
    parameter N_NEURONS      = 640,
    parameter N_STEPS        = 50,
    parameter WIDTH          = 16,
    parameter [15:0] SEED    = 16'hACE1,
    parameter THRESH_MEM_FILE = ""
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,          // pulse to begin one full encoding pass

    output reg  spike_valid,    // one-cycle pulse per valid (t,n) result
    output reg  spike_bit,
    output reg  [$clog2(N_STEPS)-1:0]   out_t,
    output reg  [$clog2(N_NEURONS)-1:0] out_n,
    output reg  done            // one-cycle pulse when the full pass completes
);

    localparam NBITS_N = $clog2(N_NEURONS);
    localparam NBITS_T = $clog2(N_STEPS);

    // ---- request-side counters ----
    reg running;
    reg issuing;                       // true while still issuing new (t,n) requests
    reg [NBITS_T-1:0] cnt_t;
    reg [NBITS_N-1:0] cnt_n;

    // ---- pipeline register: which (t,n) the in-flight LFSR/threshold read corresponds to ----
    reg              issuing_d;
    reg [NBITS_T-1:0] t_d;
    reg [NBITS_N-1:0] n_d;

    // ---- LFSR (shared, one instance, advances once per issued request) ----
    // NOTE: this instance's rst_n is tied directly to the module's rst_n.
    // A mid-operation re-seed (for running back-to-back inferences on one
    // instance without a full system reset) was originally attempted via
    // an internal async pulse, but that pulse was combinationally derived
    // from a register updated on the SAME clock edge it needed to gate --
    // a real race hazard (caught by simulation against real golden data,
    // not analysis -- see PROJECT_REPORT.md §4.8). Removed for correctness.
    // CURRENT LIMITATION: this module runs exactly one encoding pass
    // correctly per rst_n assertion. Multi-inference reseeding (without a
    // full system reset) needs a properly synchronous solution -- not yet
    // designed, tracked as a follow-up, not silently left broken.
    wire [15:0] lfsr_value;
    lfsr16 #(.SEED(SEED)) u_lfsr (
        .clk(clk),
        .rst_n(rst_n),
        .advance(issuing),
        .value(lfsr_value)
    );

    // ---- threshold ROM ----
    wire [WIDTH-1:0] threshold_data;
    threshold_mem #(
        .N_NEURONS(N_NEURONS), .WIDTH(WIDTH), .MEM_FILE(THRESH_MEM_FILE)
    ) u_thresh (
        .clk(clk),
        .addr(cnt_n),
        .data_out(threshold_data)
    );

    // one-cycle pulse that forces the LFSR back to its seed state exactly
    // when `start` is asserted, so every encoding pass begins fresh
    // (matches Python creating a new LFSR16(seed=0xACE1) per file).
    // REMOVED -- see note above u_lfsr instantiation. Kept as a comment,
    // not a dangling signal, so a future fix doesn't have to rediscover
    // why this was here.

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running        <= 1'b0;
            issuing        <= 1'b0;
            cnt_t          <= 0;
            cnt_n          <= 0;
            issuing_d      <= 1'b0;
            t_d            <= 0;
            n_d            <= 0;
            spike_valid    <= 1'b0;
            spike_bit      <= 1'b0;
            done           <= 1'b0;
        end else begin
            done <= 1'b0;

            if (start && !running) begin
                running <= 1'b1;
                issuing <= 1'b1;
                cnt_t   <= 0;
                cnt_n   <= 0;
            end else if (running) begin
                // advance the request counters if still issuing
                if (issuing) begin
                    if (cnt_n == N_NEURONS-1) begin
                        cnt_n <= 0;
                        if (cnt_t == N_STEPS-1) begin
                            issuing <= 1'b0;   // just issued the very last (t,n) request
                        end else begin
                            cnt_t <= cnt_t + 1'b1;
                        end
                    end else begin
                        cnt_n <= cnt_n + 1'b1;
                    end
                end
            end

            // pipeline: remember which (t,n) is currently in-flight
            issuing_d <= running ? issuing : 1'b0;
            t_d       <= cnt_t;
            n_d       <= cnt_n;

            // compare stage: valid one cycle after a request was issued
            if (issuing_d) begin
                spike_valid <= 1'b1;
                spike_bit   <= (lfsr_value < threshold_data) ? 1'b1 : 1'b0;
                out_t       <= t_d;
                out_n       <= n_d;
            end else begin
                spike_valid <= 1'b0;
            end

            // done pulses exactly one cycle after the very last compare was valid
            if (running && !issuing && !issuing_d) begin
                running <= 1'b0;
                done    <= 1'b1;
            end
        end
    end

endmodule
