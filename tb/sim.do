vlib work
vlog -work work ../rtl/weight_mem.v
vlog -work work ../rtl/threshold_mem.v
vlog -work work ../rtl/lfsr16.v
vlog -work work ../rtl/rate_encoder.v
vlog -work work ../rtl/aer_buffer.v
vlog -work work ../rtl/spike_gated_mac.v
vlog -work work ../rtl/v_mem.v
vlog -work work ../rtl/lif_neuron_array.v
vlog -work work ../rtl/lif_output.v
vlog -work work ../rtl/early_exit.v
vlog -work work ../rtl/snn_top.v
vlog -work work ../rtl/de2_top.v
vlog -work work tb_top.v

vsim -c -voptargs="+acc" work.tb_top
run -all
quit
