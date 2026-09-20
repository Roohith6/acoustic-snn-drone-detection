# Recompile everything cleanly
vlog -work work ../rtl/*.v
vlog -work work tb_top_8.v

# Open the wave window and clear it safely
view wave
catch {delete wave *}
vsim -voptargs="+acc" work.tb_top_8

# Core Control Signals
add wave -noupdate -divider {CONTROL PINS}
add wave -noupdate -label CLOCK /tb_top_8/uut/CLOCK_50
add wave -noupdate -label KEY_INPUTS /tb_top_8/uut/KEY
add wave -noupdate -label AUDIO_SAMPLE_ID -radix unsigned /tb_top_8/uut/sample_sel
add wave -noupdate -label START_INFERENCE /tb_top_8/uut/u_snn/start
add wave -noupdate -label DONE /tb_top_8/uut/u_snn/done

# Results
add wave -noupdate -divider {FINAL RESULTS}
add wave -noupdate -label WINNER_CLASS /tb_top_8/uut/u_snn/winner
add wave -noupdate -label AMBIENT_COUNT -radix unsigned -format Analog-Step -height 40 -max 50.0 -min 0.0 /tb_top_8/uut/u_snn/spike_cnt_ambient
add wave -noupdate -label DRONE_COUNT -radix unsigned -format Analog-Step -height 40 -max 50.0 -min 0.0 /tb_top_8/uut/u_snn/spike_cnt_drone

# Input Layer Spikes (The Actual Audio Spikes)
add wave -noupdate -divider {INPUT AUDIO SPIKES (Real 1s and 0s)}
add wave -noupdate -label IN_SPIKE_0 /tb_top_8/uut/u_snn/u_encoder/spike_bit

# Internal Neurons (The Actual Hidden Spikes)
add wave -noupdate -divider {HIDDEN NEURON SPIKES (Real 1s and 0s)}
add wave -noupdate -label NEURON_0 /tb_top_8/uut/u_snn/u_hidden/out_spike_bus[0]
add wave -noupdate -label NEURON_1 /tb_top_8/uut/u_snn/u_hidden/out_spike_bus[1]
add wave -noupdate -label NEURON_2 /tb_top_8/uut/u_snn/u_hidden/out_spike_bus[2]
add wave -noupdate -label NEURON_3 /tb_top_8/uut/u_snn/u_hidden/out_spike_bus[3]
add wave -noupdate -label NEURON_4 /tb_top_8/uut/u_snn/u_hidden/out_spike_bus[4]
add wave -noupdate -label NEURON_5 /tb_top_8/uut/u_snn/u_hidden/out_spike_bus[5]

run -all
wave zoom full
