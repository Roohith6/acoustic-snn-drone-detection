with open(r'E:\snn_2\tb\tb_top.v', 'r') as f: content = f.read()
content = content.replace('\\(\"Time', '\(\"Time')
content = content.replace('\\, uut', '\, uut')
with open(r'E:\snn_2\tb\tb_top.v', 'w') as f: f.write(content)
