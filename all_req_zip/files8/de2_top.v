// =============================================================================
// de2_top.v
// -----------------------------------------------------------------------------
// Board-level top for the DE2-115 demo. NOT used in simulation -- the
// testbenches all target snn_top.v directly. This module exists purely to
// map real board I/O onto snn_top's interface plus the new sample_sel input.
//
// Interface:
//   KEY[1:0]  -- 2-bit sample selector (see note on active-low below).
//                Held level, not a momentary press -- whatever combination
//                of the two buttons is currently held determines which of
//                4 pre-loaded audio samples gets classified.
//   LEDG[0]   -- lit if the current classification result is DRONE
//   LEDG[1]   -- lit if the current classification result is AMBIENT
//                Both LEDs stay OFF until the first classification
//                completes after power-up/reset (result_valid gate) --
//                avoids showing a misleading "ambient" LED before any
//                real inference has run.
//
// No separate "start" button: classification auto-triggers whenever the
// KEY[1:0] code changes (edge-detected), and once automatically right
// after reset so the default selection (code 00) is classified without
// needing a button press first.
//
// DE2-115 pushbuttons are ACTIVE LOW (unpressed = 1, pressed = 0). Inverted
// here so "no buttons pressed" (KEY=11 electrically) maps to sample code
// 00 -- the natural idle state selects sample 0, not sample 3.
//
// No debouncing. For a demo this is harmless: a bouncy transition may
// cause classification to auto-retrigger once or twice in quick
// succession, ending on the same correct result a few tens of
// milliseconds later -- not worth the added complexity given the time
// constraint this was built under. Worth adding if this becomes a
// longer-lived board design.
// =============================================================================
module de2_top (
    input  wire        CLOCK_50,
    input  wire [1:0]  KEY,        // KEY[1:0], active-low pushbuttons
    output wire [1:0]  LEDG        // LEDG[0]=drone, LEDG[1]=ambient
);

    wire clk = CLOCK_50;

    // ---- reset ----
    // No dedicated reset button was specified in the requested interface
    // (only KEY[1:0] for selection). Using a simple power-on reset
    // generator here instead of requiring a third button -- holds rst_n
    // low for a handful of cycles after configuration, then releases.
    // If a manual reset button is preferred instead, wire rst_n to a
    // spare KEY (e.g. KEY[3] on the physical connector, even though only
    // KEY[1:0] are used for selection here) and drop this counter.
    reg [3:0] por_cnt = 4'd0;
    reg       rst_n_r = 1'b0;
    always @(posedge clk) begin
        if (por_cnt != 4'hF) begin
            por_cnt <= por_cnt + 1'b1;
            rst_n_r <= 1'b0;
        end else begin
            rst_n_r <= 1'b1;
        end
    end
    wire rst_n = rst_n_r;

    // ---- sample selector, active-low inversion ----
    wire [1:0] sample_sel = ~KEY;

    // ---- auto-trigger classification on selector change (or once after reset) ----
    reg [1:0] sample_sel_d;
    reg       first_run_done;
    reg       start_pulse;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_sel_d   <= 2'b00;
            first_run_done <= 1'b0;
            start_pulse    <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (!first_run_done) begin
                // fire once automatically right after reset, classifying
                // whatever sample_sel currently reads (default code 00
                // if no buttons are held)
                start_pulse    <= 1'b1;
                first_run_done <= 1'b1;
                sample_sel_d   <= sample_sel;
            end else if (sample_sel != sample_sel_d) begin
                start_pulse  <= 1'b1;
                sample_sel_d <= sample_sel;
            end
        end
    end

    // ---- core ----
    wire done, winner;
    wire signed [63:0] z2_total0, z2_total1; // unused on the board, kept for debug/simulation parity

    snn_top #(
        // NOTE: fill these in with real paths relative to your Quartus
        // project directory before compiling -- these are placeholders
        // matching the file NAMES already generated and verified, not
        // necessarily your project's actual folder layout.
        .THRESH_MEM_FILE_0("threshold_ambient.mem"),   // code 00 -- verified sample
        .THRESH_MEM_FILE_1("threshold_drone.mem"),     // code 01 -- verified sample
        .THRESH_MEM_FILE_2("threshold_ambient2.mem"),  // code 10 -- verified sample
        .THRESH_MEM_FILE_3("threshold_drone2.mem"),    // code 11 -- verified sample
        .W1_MEM_FILE("W1_folded_input_hidden.mem"),
        .B1_MEM_FILE("b1_folded_hidden.mem"),
        .SIGMOID_MEM_FILE("sigmoid_lut.mem"),
        .W2_MEM_FILE("W2_hidden_output.mem"),
        .B2_MEM_FILE("b2_output.mem")
    ) u_snn (
        .clk(clk), .rst_n(rst_n),
        .start(start_pulse), .sample_sel(sample_sel),
        .done(done), .winner(winner),
        .z2_total0(z2_total0), .z2_total1(z2_total1)
    );

    // ---- result_valid gate: LEDs stay off until first real result ----
    reg result_valid;
    reg latched_winner;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_valid   <= 1'b0;
            latched_winner <= 1'b0;
        end else if (done) begin
            result_valid   <= 1'b1;
            latched_winner <= winner;
        end
    end

    assign LEDG[0] = result_valid &&  latched_winner; // drone
    assign LEDG[1] = result_valid && ~latched_winner; // ambient

endmodule
