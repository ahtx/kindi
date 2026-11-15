# SPDX-License-Identifier: AGPL-3.0-only
"""Read DXGI adapter budgets in the calling process; never change reservations."""
import ctypes
from ctypes import wintypes
import json
import sys
import uuid


class GUID(ctypes.Structure):
    _fields_=[('data',ctypes.c_ubyte*16)]
    def __init__(self,value):
        super().__init__()
        self.data[:]=uuid.UUID(value).bytes_le


class LUID(ctypes.Structure):
    _fields_=[('low',wintypes.DWORD),('high',wintypes.LONG)]


class Description(ctypes.Structure):
    _fields_=[('description',wintypes.WCHAR*128),('vendor',wintypes.UINT),('device',wintypes.UINT),
              ('subsystem',wintypes.UINT),('revision',wintypes.UINT),
              ('dedicated_video',ctypes.c_size_t),('dedicated_system',ctypes.c_size_t),
              ('shared_system',ctypes.c_size_t),('luid',LUID),('flags',wintypes.UINT)]


class MemoryInfo(ctypes.Structure):
    _fields_=[('budget',ctypes.c_uint64),('current_usage',ctypes.c_uint64),
              ('available_for_reservation',ctypes.c_uint64),('current_reservation',ctypes.c_uint64)]


def method(obj,index,result,*arguments):
    table=ctypes.cast(obj,ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result,ctypes.c_void_p,*arguments)(table[index])


def release(obj):
    if obj:
        method(obj,2,wintypes.ULONG)(obj)


def snapshot():
    if sys.platform!='win32':
        return {'error':'DXGI requires native Windows'}
    factory=ctypes.c_void_p()
    dxgi=ctypes.WinDLL('dxgi')
    create=dxgi.CreateDXGIFactory1
    create.argtypes=[ctypes.POINTER(GUID),ctypes.POINTER(ctypes.c_void_p)]
    create.restype=wintypes.LONG
    iid=GUID('770aae78-f26f-4dba-a829-253c83d1b387')
    status=create(ctypes.byref(iid),ctypes.byref(factory))
    if status:
        return {'error':f'CreateDXGIFactory1:0x{status&0xffffffff:08x}'}
    rows=[]
    try:
        enum=method(factory,12,wintypes.LONG,wintypes.UINT,ctypes.POINTER(ctypes.c_void_p))
        for index in range(16):
            adapter=ctypes.c_void_p()
            status=enum(factory,index,ctypes.byref(adapter))
            if status:
                break
            adapter3=ctypes.c_void_p()
            try:
                description=Description()
                status=method(adapter,10,wintypes.LONG,ctypes.POINTER(Description))(adapter,ctypes.byref(description))
                if status:
                    continue
                row={'name':description.description,'vendor':description.vendor,'device':description.device,
                     'dedicated_video_bytes':description.dedicated_video}
                aid=GUID('645967a4-1392-4310-a798-8053ce3e93fd')
                status=method(adapter,0,wintypes.LONG,ctypes.POINTER(GUID),ctypes.POINTER(ctypes.c_void_p))(
                    adapter,ctypes.byref(aid),ctypes.byref(adapter3))
                if status:
                    row['error']=f'IDXGIAdapter3:0x{status&0xffffffff:08x}'
                else:
                    query=method(adapter3,14,wintypes.LONG,wintypes.UINT,wintypes.UINT,ctypes.POINTER(MemoryInfo))
                    for group,name in [(0,'local'),(1,'nonlocal')]:
                        info=MemoryInfo()
                        status=query(adapter3,0,group,ctypes.byref(info))
                        row[name]={'error':f'QueryVideoMemoryInfo:0x{status&0xffffffff:08x}'} if status else {
                            field:getattr(info,field) for field,_ in info._fields_}
                rows.append(row)
            finally:
                release(adapter3)
                release(adapter)
    finally:
        release(factory)
    return {'scope':'DXGI budget/usage for calling process; not whole-adapter free VRAM','adapters':rows}


if __name__=='__main__':
    print(json.dumps(snapshot()),flush=True)


def available_local_bytes(gpu_name, log=print):
    """Available bytes within this process's local budget, not global free VRAM."""
    report = snapshot()
    if 'error' in report:
        raise RuntimeError('Windows GPU budget query failed: ' + report['error'])
    matches = [adapter for adapter in report['adapters']
               if adapter['name'].strip() == gpu_name.strip()]
    if not matches:
        raise RuntimeError('Windows GPU budget adapter was not found: ' + gpu_name)
    values = []
    for adapter in matches:
        info = adapter.get('local', {})
        if 'error' in info or info.get('budget', 0) <= 0 or info.get('current_usage', 0) <= 0:
            raise RuntimeError('Windows GPU local budget/usage is unavailable')
        values.append(info)
    # Duplicate adapter descriptions are handled conservatively. This control
    # is restricted to one HIP card by the engine before loading weights.
    info = min(values, key=lambda item: item['budget'] - item['current_usage'])
    available = max(0, info['budget'] - info['current_usage'])
    log('Windows local budget: %.1f MiB, current usage %.1f MiB, available %.1f MiB' %
        (info['budget']/2**20, info['current_usage']/2**20, available/2**20))
    return available
