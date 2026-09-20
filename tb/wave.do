vsim -voptargs="+acc" work.tb_top
add wave -position insertpoint sim:/tb_top/uut/*
run -all
wave zoom full
