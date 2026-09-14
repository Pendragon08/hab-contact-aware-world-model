import pathlib,time
p=pathlib.Path(__file__).parent/"counter"
i=0
while True:
 p.write_text(str(i));i+=1;time.sleep(.1)
