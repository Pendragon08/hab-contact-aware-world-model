import pathlib,time,os
p=pathlib.Path(__file__).parent
(p/'child.pid').write_text(str(os.getpid()))
for i in range(15):
 (p/'counter').write_text(str(i))
 time.sleep(.2)
(p/'child.complete').write_text('done')
