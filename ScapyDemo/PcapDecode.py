#!/usr/bin/env python  
# encoding: utf-8  
"""
@author: Alfons
@contact: alfons_xh@163.com
@file: PcapDecode.py 
@time: 2018/1/18 12:05 
@version: v1.0 
"""
import sys
sys.setrecursionlimit(10000)
from scapy.all import rdpcap
import scapy_http.http


packets = rdpcap('feel.pcap')
for p in packets:
    print '=' * 78
#     str_a = p.show()
# pass