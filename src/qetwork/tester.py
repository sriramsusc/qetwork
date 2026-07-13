from qetwork.kernel.timeline import Timeline
t = Timeline()
def tick(n):
    print('tick', n, 'at time', t.now())
    if n < 3:
        t.schedule(tick, n + 1, at=t.now(), delay=100)
t.schedule(tick, 1, at=0)
t.run()
print('final time', t.now())
