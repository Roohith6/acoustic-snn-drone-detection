// =============================================================================
// output_layer.v
// -----------------------------------------------------------------------------
// Unlike the input layer, a1 is a single continuous value per hidden neuron
// (already time-integrated across all 50 steps inside hidden_neuron.v), not
// a spike train -- so this is a plain dense 128-wide dot product, computed
// once per sample, not spike-gated and not per-timestep.
//
//   z2[o] = sum_h a1[h] * W2[h,o] + b2[o]      (o = 0 ambient, 1 drone)
//   predicted class = argmax(z2)                (softmax skipped -- monotonic)
//
// a1 is unsigned Q0.16 (from sigmoid_lut.v); W2/b2 are signed int16 with
// their OWN, different scale factors (same pattern as combine_scale.v).
// Only the argmax matters for the final decision, so z2 is kept in raw
// mixed-scale integer units -- no need to descale to a real-valued
// probability, just a consistent common scale for the two outputs so
// their relative order is preserved.
//
// Streaming interface: caller presents one (h, a1_value) pair at a time
// with h_valid; module reads both W2[h,0] and W2[h,1] and accumulates.
// After all N_POST pairs, pulse `finalize` to add the (rescaled) bias and
// compute the argmax.
// =============================================================================
module output_layer #(
    parameter N_POST = 128,
    parameter N_OUT  = 2,
    parameter W2_MEM_FILE = "",
    parameter B2_MEM_FILE = "",
    parameter signed [31:0] C_FIXED = 32'd817345198,  // 65535*W2_scale/b2_scale, Q?.20
    parameter SHIFT = 20
) (
    input  wire clk,
    input  wire rst_n,

    input  wire        start,       // clear accumulators
    input  wire [$clog2(N_POST)-1:0] h_index,
    input  wire [15:0] a1_value,    // unsigned Q0.16
    input  wire        h_valid,     // pulse: consume (h_index, a1_value)
    output reg          busy,        // high while processing this h

    input  wire         finalize,    // pulse after all N_POST pairs consumed
    output reg           done,        // pulse when argmax is ready
    output reg           winner,      // 0 = ambient, 1 = drone
    output reg  signed [63:0] z2_total0,
    output reg  signed [63:0] z2_total1
);

    reg signed [63:0] acc0, acc1;

    // 2-deep weight read pipeline (W2 has NUM_POST=2: addr = h*2+o)
    wire [$clog2(N_POST*N_OUT)-1:0] w_addr0 = h_index * N_OUT + 0;
    wire [$clog2(N_POST*N_OUT)-1:0] w_addr1 = h_index * N_OUT + 1;
    wire signed [15:0] w2_data0, w2_data1;

    weight_mem #(.NUM_PRE(N_POST), .NUM_POST(N_OUT), .WIDTH(16), .MEM_FILE(W2_MEM_FILE))
        u_w2_0 (.clk(clk), .addr(w_addr0), .data_out(w2_data0));
    weight_mem #(.NUM_PRE(N_POST), .NUM_POST(N_OUT), .WIDTH(16), .MEM_FILE(W2_MEM_FILE))
        u_w2_1 (.clk(clk), .addr(w_addr1), .data_out(w2_data1));

    reg signed [15:0] b2_int0, b2_int1;
    initial begin
        // small fixed 2-entry bias table -- simplest correct approach for
        // just 2 values (no need for a generic memory module here)
        if (B2_MEM_FILE != "") begin
            $readmemh(B2_MEM_FILE, b2_lut);
        end
    end
    reg signed [15:0] b2_lut [0:1];
    always @(*) begin
        b2_int0 = b2_lut[0];
        b2_int1 = b2_lut[1];
    end

    reg pending;
    reg signed [15:0] a1_signed_ext_pending;
    reg [$clog2(N_POST)-1:0] h_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc0 <= 0; acc1 <= 0;
            busy <= 1'b0;
            pending <= 1'b0;
            done <= 1'b0;
            winner <= 1'b0;
        end else begin
            done <= 1'b0;

            if (start) begin
                acc0 <= 0; acc1 <= 0;
            end

            if (h_valid && !busy) begin
                busy <= 1'b1;
                pending <= 1'b1;
            end else begin
                pending <= 1'b0;
            end

            if (pending) begin
                // w2_data0/1 now valid for the h_index that was presented last cycle
                acc0 <= acc0 + ($signed({1'b0, a1_value}) * w2_data0);
                acc1 <= acc1 + ($signed({1'b0, a1_value}) * w2_data1);
                busy <= 1'b0;
            end

            if (finalize) begin
                z2_total0 <= acc0 + (($signed(b2_int0) * C_FIXED) >>> SHIFT);
                z2_total1 <= acc1 + (($signed(b2_int1) * C_FIXED) >>> SHIFT);
                winner    <= ((acc1 + (($signed(b2_int1) * C_FIXED) >>> SHIFT)) >
                              (acc0 + (($signed(b2_int0) * C_FIXED) >>> SHIFT))) ? 1'b1 : 1'b0;
                done <= 1'b1;
            end
        end
    end

endmodule
