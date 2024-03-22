from jqdatasdk import *
auth('18372177165','Jty001105')

df0 = get_bars('000300.XSHG', 11760, unit='5m',
               fields=['volume', 'money', 'open', 'low', 'close', 'high'], include_now=True, end_dt='2021-12-31')