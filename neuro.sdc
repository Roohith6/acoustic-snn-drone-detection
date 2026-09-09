# =============================================================================
# neuro.sdc
# -----------------------------------------------------------------------------
# Without this file, TimeQuest has no clock constraint and silently defaults
# to derive_clocks -period 1.0 (1GHz) -- an unconstrained, unrealistic check
# that fails for essentially any nontrivial design and tells you nothing
# useful. This file gives it a real target: 50MHz, matching the DE2-115's
# onboard oscillator (pin PIN_Y2 in the standard DE2-115 pin assignment --
# double check against your board's actual .qsf / pin planner if this
# doesn't match).
#
# Add this file to the Quartus project directory (same folder as the .qpf),
# then Assignments -> Settings -> TimeQuest Timing Analyzer -> make sure
# it's picked up (Quartus auto-detects a same-named .sdc in 13.x), or add it
# explicitly via Assignments -> Settings -> Timing Analyzer if needed.
# Recompile after adding this -- the TimeQuest report will then reflect a
# real target instead of the meaningless 1GHz default.
# =============================================================================

create_clock -name clk -period 20.000 [get_ports {clk}]

derive_clock_uncertainty
