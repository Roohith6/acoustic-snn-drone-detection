vsim -c -voptargs="+acc" work.tb_top
run 2ms
examine -radix dec /tb_top/uut/u_snn/u_output/v_0
examine -radix dec /tb_top/uut/u_snn/u_output/v_1
examine -radix dec /tb_top/uut/u_snn/u_output/cnt_0
examine -radix dec /tb_top/uut/u_snn/u_output/cnt_1
examine -radix dec /tb_top/uut/u_snn/u_output/v_thresh
examine -radix dec /tb_top/uut/u_snn/u_output/bias_mem(0)
examine -radix dec /tb_top/uut/u_snn/u_output/bias_mem(1)
examine -radix unsigned /tb_top/uut/u_snn/u_output/state
examine -radix unsigned /tb_top/uut/u_snn/u_output/u_mac/busy
quit
